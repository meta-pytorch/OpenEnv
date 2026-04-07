from __future__ import annotations

from pathlib import Path
import runpy


def _emit_startup_failure(exc: Exception) -> None:
    error_str = str(exc).replace(" ", "_")
    print(f"[STEP] step=0 action=startup reward=0.00 done=true error={error_str}", flush=True)
    print("FINAL_AVG_SCORE=0.000", flush=True)


def _load_env_main():
    env_inference = Path(__file__).resolve().parent / "envs" / "email_triage_env" / "inference.py"
    if not env_inference.exists():
        raise FileNotFoundError(f"missing_env_inference_file:{env_inference}")

    module_globals = runpy.run_path(str(env_inference))
    env_main = module_globals.get("main")
    if not callable(env_main):
        raise RuntimeError("env_main_not_found")
    return env_main


def main() -> None:
    try:
        env_main = _load_env_main()
    except Exception as exc:
        _emit_startup_failure(exc)
        return

    try:
        env_main()
    except Exception as exc:
        _emit_startup_failure(exc)


if __name__ == "__main__":
    main()
