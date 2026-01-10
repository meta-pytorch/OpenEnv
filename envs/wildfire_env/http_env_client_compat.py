"""
Compatibility shim for HTTPEnvClient when openenv-core package doesn't have it.

This file provides HTTPEnvClient as a workaround until the openenv-core package
is updated with the http_env_client module.
"""

try:
    # Try to import from openenv_core (should work once package is updated)
    from openenv_core.http_env_client import HTTPEnvClient
except ImportError:
    # Fallback: create HTTPEnvClient from EnvClient
    # Try openenv.core first (correct package structure)
    try:
        from openenv.core.env_client import EnvClient
        from openenv.core.env_server.types import State
    except ImportError:
        # Fallback to openenv_core (legacy compatibility)
        from openenv_core.env_client import EnvClient
        from openenv_core.env_server.types import State
    
    from typing import Generic, TypeVar

    ActT = TypeVar("ActT")
    ObsT = TypeVar("ObsT")

    class HTTPEnvClient(EnvClient[ActT, ObsT, State], Generic[ActT, ObsT]):
        """
        HTTP Environment Client compatibility shim.
        
        This is a wrapper around EnvClient that uses the standard State type
        and only requires 2 type parameters (action and observation types).
        """
        pass

__all__ = ["HTTPEnvClient"]

