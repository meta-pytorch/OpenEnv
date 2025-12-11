# Copyright (c) Yogesh Singla and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for Julia Environment with Request Queuing and Backpressure.

NEW FEATURES:
- Request queue with maximum size for backpressure
- Immediate failure when queue is full (no timeout waiting)
- Queue metrics and monitoring
- Enhanced health endpoint with detailed metrics
- Worker status reporting

Features:
- Async Julia code execution
- Request queuing with backpressure
- Worker health monitoring
- Automatic error recovery
- Comprehensive logging and metrics
"""

import asyncio
import logging
import os
import sys
import traceback
from asyncio import Queue, QueueFull
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from core.tools import JuliaExecutor
from ..models import JuliaAction, JuliaObservation
from .julia_codeact_env import JuliaCodeActEnv

# Configuration
MAX_WORKERS = int(os.getenv("JULIA_MAX_WORKERS", "64"))
MAX_QUEUE_SIZE = int(os.getenv("JULIA_MAX_QUEUE_SIZE", "100"))  # NEW: Request queue limit
ENABLE_WEB = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")
EXECUTION_TIMEOUT = int(os.getenv("JULIA_EXECUTION_TIMEOUT", "120"))
LOG_FILE = os.getenv("JULIA_LOG_FILE", "/tmp/run.log")
LOG_LEVEL = os.getenv("JULIA_LOG_LEVEL", "INFO")

# Global resources
executor = None
request_queue = None

# Metrics
total_requests = 0
queued_requests = 0
rejected_requests = 0
completed_requests = 0
failed_requests = 0


def setup_logging():
    """Configure logging."""
    logger = logging.getLogger("julia_env")
    logger.setLevel(getattr(logging, LOG_LEVEL))

    if logger.handlers:
        return logger

    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(process)d:%(thread)d] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file {LOG_FILE}: {e}")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager with request queue initialization."""
    global executor, request_queue

    logger.info("=" * 80)
    logger.info("Starting Julia Environment Server with Request Queuing")
    logger.info(f"Max Workers: {MAX_WORKERS}")
    logger.info(f"Max Queue Size: {MAX_QUEUE_SIZE}")
    logger.info(f"Execution Timeout: {EXECUTION_TIMEOUT}s")
    logger.info(f"Log File: {LOG_FILE}")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info("=" * 80)

    try:
        # Initialize request queue with backpressure
        request_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        logger.info(f"✅ Request queue created (max_size={MAX_QUEUE_SIZE})")

        # Create thread pool
        executor = ThreadPoolExecutor(
            max_workers=MAX_WORKERS, thread_name_prefix="julia_worker"
        )
        logger.info(f"✅ Thread pool created with {MAX_WORKERS} workers")

        # Initialize Julia process pool with enhanced monitoring
        pool_size = MAX_WORKERS
        logger.info(f"🔧 Initializing Julia process pool with {pool_size} workers...")

        pool_enabled = JuliaExecutor.enable_process_pool(
            size=pool_size,
            timeout=EXECUTION_TIMEOUT,
        )

        if pool_enabled:
            logger.info(
                f"✅ Julia process pool initialized with {pool_size} workers "
                f"(health monitoring enabled)"
            )
        else:
            logger.warning("⚠️ Julia process pool not available, using standard execution")

        logger.info(f"✅ Julia Environment Server started successfully")
        print(f"✅ Julia Server: {MAX_WORKERS} workers, {MAX_QUEUE_SIZE} queue limit")

    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        logger.error(traceback.format_exc())
        raise

    yield

    # Shutdown
    logger.info("Shutting down Julia Environment Server...")
    try:
        JuliaExecutor.shutdown_pool()
        logger.info("✅ Julia process pool shut down")

        executor.shutdown(wait=True, cancel_futures=False)
        logger.info("✅ All workers completed gracefully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    logger.info("✅ Julia Environment Server shutdown complete")
    print("✅ Julia Environment Server shutdown complete")


app = FastAPI(
    title="Julia Environment Server",
    description="Async Julia code execution with request queuing and backpressure",
    version="3.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    error_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.error(f"[ERROR-{error_id}] Uncaught exception in {request.url.path}")
    logger.error(f"[ERROR-{error_id}] Request: {request.method} {request.url}")
    logger.error(f"[ERROR-{error_id}] Exception: {type(exc).__name__}: {exc}")
    logger.error(f"[ERROR-{error_id}] Traceback:\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "type": type(exc).__name__,
            "message": str(exc),
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
        },
    )


async def execute_julia_async(
    action: JuliaAction, request_id: str = None
) -> JuliaObservation:
    """
    Execute Julia code asynchronously with timeout and error recovery.

    This queues the request and executes it when a worker is available.

    Note: Timeouts are NOT retried as they indicate infinite loops or
    extremely long-running code that won't benefit from retries.
    """
    global completed_requests, failed_requests

    if request_id is None:
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    loop = asyncio.get_event_loop()
    max_retries = 2
    retry_count = 0

    logger.info(
        f"[{request_id}] Starting Julia execution "
        f"(timeout: {EXECUTION_TIMEOUT}s, core_code: {len(action.core_code)} chars)"
    )

    execution_start = datetime.now()

    while retry_count <= max_retries:
        env = None
        try:
            env = JuliaCodeActEnv()

            observation = await asyncio.wait_for(
                loop.run_in_executor(executor, env.step, action),
                timeout=EXECUTION_TIMEOUT,
            )

            elapsed = (datetime.now() - execution_start).total_seconds()
            logger.info(
                f"[{request_id}] Julia execution completed in {elapsed:.2f}s "
                f"(tests_passed={observation.tests_passed}, "
                f"tests_failed={observation.tests_failed}, "
                f"reward={observation.reward})"
            )

            completed_requests += 1
            return observation

        except asyncio.TimeoutError:
            # Don't retry timeouts - they indicate infinite loops or extremely long code
            # Retrying won't help and wastes 4+ minutes
            elapsed = (datetime.now() - execution_start).total_seconds()
            logger.warning(
                f"[{request_id}] Julia execution timeout after {elapsed:.2f}s"
            )
            logger.error(
                f"[{request_id}] Execution exceeded timeout limit (not retrying timeouts)"
            )
            failed_requests += 1
            return JuliaObservation(
                stdout="",
                stderr=f"Execution timeout after {EXECUTION_TIMEOUT}s",
                exit_code=124,  # Standard timeout exit code
                tests_passed=0,
                tests_failed=1,
                code_compiles=False,
                reward=0.0,
                done=True,
            )

        except Exception as e:
            retry_count += 1
            elapsed = (datetime.now() - execution_start).total_seconds()
            logger.error(
                f"[{request_id}] Julia execution error after {elapsed:.2f}s "
                f"(attempt {retry_count}/{max_retries + 1}): {type(e).__name__}: {e}"
            )
            logger.error(f"[{request_id}] Traceback:\n{traceback.format_exc()}")

            if retry_count > max_retries:
                logger.error(
                    f"[{request_id}] Julia execution failed permanently after {max_retries + 1} attempts"
                )
                failed_requests += 1
                return JuliaObservation(
                    stdout="",
                    stderr=f"Execution error: {str(e)}",
                    exit_code=-1,
                    tests_passed=0,
                    tests_failed=1,
                    code_compiles=False,
                    reward=0.0,
                    done=True,
                )

            await asyncio.sleep(0.5)

        finally:
            if env is not None:
                try:
                    del env
                except Exception as cleanup_error:
                    logger.debug(f"[{request_id}] Cleanup warning: {cleanup_error}")


@app.post("/reset")
async def reset(request: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Reset endpoint - returns initial observation."""
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info(f"[{request_id}] Reset request received")

    try:
        loop = asyncio.get_event_loop()
        env = JuliaCodeActEnv()
        observation = await asyncio.wait_for(
            loop.run_in_executor(executor, env.reset),
            timeout=30.0,
        )

        obs_dict = asdict(observation)
        reward = obs_dict.pop("reward", None)
        done = obs_dict.pop("done", False)
        obs_dict.pop("metadata", None)

        logger.info(f"[{request_id}] Reset completed successfully")

        return {
            "observation": obs_dict,
            "reward": reward,
            "done": done,
        }
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Reset timeout")
        raise HTTPException(status_code=504, detail="Reset operation timed out")
    except Exception as e:
        logger.error(f"[{request_id}] Reset error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step endpoint with request queuing and backpressure.

    NEW BEHAVIOR:
    - Checks if queue is full before accepting request
    - Returns immediate error (429 Too Many Requests) if queue is full
    - No timeout waiting for workers
    """
    global total_requests, queued_requests, rejected_requests

    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    total_requests += 1

    try:
        action_data = request.get("action", {})
        if not action_data:
            logger.warning(f"[{request_id}] Step request with empty action")
            raise HTTPException(status_code=400, detail="Action data is required")

        metadata = action_data.pop("metadata", {})
        action = JuliaAction(**action_data)
        action.metadata = metadata

        # CHECK QUEUE SIZE - BACKPRESSURE MECHANISM
        current_queue_size = request_queue.qsize()

        if current_queue_size >= MAX_QUEUE_SIZE:
            # IMMEDIATE REJECTION - Don't wait
            rejected_requests += 1
            logger.error(
                f"[{request_id}] REQUEST REJECTED - Queue is FULL "
                f"({current_queue_size}/{MAX_QUEUE_SIZE}). "
                f"All workers are busy with pending requests."
            )

            # Return 429 Too Many Requests with backpressure info
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Request queue full",
                    "message": f"Server is overloaded. Queue size: {current_queue_size}/{MAX_QUEUE_SIZE}. "
                               f"All {MAX_WORKERS} workers are busy. Please retry later.",
                    "queue_size": current_queue_size,
                    "max_queue_size": MAX_QUEUE_SIZE,
                    "workers_busy": MAX_WORKERS,
                    "retry_after_seconds": 10,  # Suggest retry after 10 seconds
                },
            )

        # Queue has space - add request
        try:
            # Use put_nowait to avoid blocking (should never block since we checked size)
            request_queue.put_nowait((request_id, action))
            queued_requests += 1

            logger.info(
                f"[{request_id}] Step request queued "
                f"(queue_size={current_queue_size + 1}/{MAX_QUEUE_SIZE})"
            )

        except QueueFull:
            # Race condition - queue filled up between check and put
            rejected_requests += 1
            logger.error(f"[{request_id}] REQUEST REJECTED - Queue filled (race condition)")

            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Request queue full (race condition)",
                    "message": f"Server is overloaded. Please retry later.",
                    "retry_after_seconds": 10,
                },
            )

        # Execute request (worker will pick it up from queue)
        observation = await execute_julia_async(action, request_id)

        # Remove from queue after completion
        try:
            request_queue.get_nowait()
        except:
            pass  # Queue might have been processed already

        # Serialize observation
        obs_dict = asdict(observation)
        reward = obs_dict.pop("reward", None)
        done = obs_dict.pop("done", False)
        obs_dict.pop("metadata", None)

        logger.info(
            f"[{request_id}] Step completed - reward={reward}, "
            f"tests_passed={observation.tests_passed}, tests_failed={observation.tests_failed}"
        )

        return {
            "observation": obs_dict,
            "reward": reward,
            "done": done,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Step endpoint error: {e}")
        logger.error(f"[{request_id}] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Step execution failed: {str(e)}")


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """State endpoint with queue and worker metrics."""
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        # Get Julia pool metrics if available
        pool_metrics = {}
        try:
            pool_metrics = JuliaExecutor.get_pool_metrics()
        except:
            pass

        return {
            "max_workers": MAX_WORKERS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "current_queue_size": request_queue.qsize() if request_queue else 0,
            "executor_type": "ThreadPoolExecutor",
            "status": "ready",
            "timeout": EXECUTION_TIMEOUT,
            "log_file": LOG_FILE,
            "memory_mb": memory_info.rss / 1024 / 1024,
            "threads": len(process.threads()),
            "metrics": {
                "total_requests": total_requests,
                "queued_requests": queued_requests,
                "rejected_requests": rejected_requests,
                "completed_requests": completed_requests,
                "failed_requests": failed_requests,
                "rejection_rate": (
                    rejected_requests / total_requests if total_requests > 0 else 0
                ),
            },
            "pool_metrics": pool_metrics,
        }
    except ImportError:
        return {
            "max_workers": MAX_WORKERS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "current_queue_size": request_queue.qsize() if request_queue else 0,
            "executor_type": "ThreadPoolExecutor",
            "status": "ready",
            "timeout": EXECUTION_TIMEOUT,
            "log_file": LOG_FILE,
            "metrics": {
                "total_requests": total_requests,
                "queued_requests": queued_requests,
                "rejected_requests": rejected_requests,
                "completed_requests": completed_requests,
                "failed_requests": failed_requests,
            },
        }
    except Exception as e:
        logger.warning(f"Could not get full state info: {e}")
        return {
            "max_workers": MAX_WORKERS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "current_queue_size": request_queue.qsize() if request_queue else 0,
            "executor_type": "ThreadPoolExecutor",
            "status": "ready",
        }


@app.get("/health")
async def health() -> Dict[str, str]:
    """
    Enhanced health check with worker and queue status.
    """
    try:
        if executor is None or request_queue is None:
            logger.error("Health check failed: resources not initialized")
            raise HTTPException(status_code=503, detail="Service not ready")

        queue_size = request_queue.qsize()
        queue_utilization = queue_size / MAX_QUEUE_SIZE if MAX_QUEUE_SIZE > 0 else 0

        # Get pool health
        pool_health = "unknown"
        try:
            pool_metrics = JuliaExecutor.get_pool_metrics()
            available_workers = pool_metrics.get('available_workers', 0)
            pool_health = "healthy" if available_workers > 0 else "degraded"
        except:
            pass

        health_status = "healthy"
        if queue_utilization > 0.9:
            health_status = "degraded"  # Queue is >90% full
        if pool_health == "degraded":
            health_status = "degraded"

        return {
            "status": health_status,
            "workers": str(MAX_WORKERS),
            "queue_size": str(queue_size),
            "max_queue_size": str(MAX_QUEUE_SIZE),
            "queue_utilization": f"{queue_utilization:.2%}",
            "pool_health": pool_health,
            "timeout": str(EXECUTION_TIMEOUT),
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """
    Enhanced metrics endpoint with detailed worker and queue metrics.
    """
    try:
        pool_metrics = {}
        try:
            pool_metrics = JuliaExecutor.get_pool_metrics()
        except:
            pass

        return {
            "timestamp": datetime.now().isoformat(),
            "server": {
                "max_workers": MAX_WORKERS,
                "max_queue_size": MAX_QUEUE_SIZE,
                "execution_timeout": EXECUTION_TIMEOUT,
            },
            "requests": {
                "total": total_requests,
                "queued": queued_requests,
                "rejected": rejected_requests,
                "completed": completed_requests,
                "failed": failed_requests,
                "rejection_rate": (
                    rejected_requests / total_requests if total_requests > 0 else 0
                ),
                "success_rate": (
                    completed_requests / (completed_requests + failed_requests)
                    if (completed_requests + failed_requests) > 0 else 0
                ),
            },
            "queue": {
                "current_size": request_queue.qsize() if request_queue else 0,
                "max_size": MAX_QUEUE_SIZE,
                "utilization": (
                    request_queue.qsize() / MAX_QUEUE_SIZE if MAX_QUEUE_SIZE > 0 else 0
                ),
            },
            "pool": pool_metrics,
        }
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


def main():
    """Main entry point for running the server."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
