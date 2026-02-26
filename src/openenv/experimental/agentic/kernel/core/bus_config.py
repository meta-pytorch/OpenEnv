# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""AgentBus configuration utilities."""

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse


@dataclass
class AgentBusOptions:
    """Configuration options for AgentBus integration in the runtime.

    When host/port are None, AgentBusHelper will auto-start an in-process server.

    local_port: Pin the localhost port for the agentbus gRPC server.
        Applies to all backends: memory://, zippydb://, and http(s):// (relay).
        If None, a port is auto-assigned (bind to 0).
    """

    bus_id: str
    host: Optional[str] = None
    port: Optional[int] = None
    disable_safety: bool = False
    run_voter: bool = False
    backend: str = "memory"
    zippy_use_case_id: Optional[int] = None
    local_port: Optional[int] = None


def parse_agentbus_url(url: str) -> dict[str, Any]:
    """Parse an agentbus URL into connection parameters.

    Uses urllib.parse for robust parsing (handles IPv6, etc.).

    Supported formats:
        memory:///bus_id           - In-process server with in-memory storage
        zippydb://<id>/bus_id      - Direct ZippyDB connection
        http://<host:port>/bus_id  - Connect to an existing gRPC server
        https://<host:port>/bus_id - Connect to an existing gRPC server (TLS)

    Port is no longer accepted in memory:// URLs. Use --agentbus-port instead.

    Returns:
        dict with keys: host, port, backend, bus_id, zippy_use_case_id (as applicable)

    Raises:
        ValueError: If the URL format is invalid
    """
    parsed = urlparse(url)

    if parsed.scheme == "memory":
        result: dict[str, Any] = {"backend": "memory"}
        if parsed.netloc:
            raise ValueError(
                f"Port in memory:// URL is no longer supported. "
                f"Use --agentbus-port to specify the port. Got: '{url}'"
            )
        bus_id = parsed.path.lstrip("/")
        if bus_id:
            result["bus_id"] = bus_id
        return result

    if parsed.scheme == "zippydb":
        rest = parsed.netloc or parsed.path
        if not rest:
            raise ValueError("zippydb:// requires a use case ID, e.g. zippydb://54451")
        try:
            use_case_id = int(rest)
        except ValueError:
            raise ValueError(
                f"Invalid ZippyDB use case ID: '{rest}'. Must be an integer."
            )
        result: dict[str, Any] = {
            "backend": "zippydb",
            "zippy_use_case_id": use_case_id,
        }
        bus_id = parsed.path.lstrip("/")
        if bus_id:
            result["bus_id"] = bus_id
        return result

    if parsed.scheme in ("http", "https"):
        if not parsed.hostname:
            raise ValueError(
                f"{parsed.scheme}:// requires host and port, e.g. http://localhost:9999"
            )
        if parsed.port is None:
            raise ValueError(f"Invalid address: '{parsed.netloc}'. Expected host:port.")
        result: dict[str, Any] = {"host": parsed.hostname, "port": parsed.port}
        bus_id = parsed.path.lstrip("/")
        if bus_id:
            result["bus_id"] = bus_id
        return result

    raise ValueError(
        f"Invalid agentbus URL: '{url}'. "
        "Expected memory://, zippydb://<use_case_id>, http://<host:port>, or https://<host:port>"
    )
