# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from dataclasses import dataclass


@dataclass
class AgentBusConfig:
    """
    Configuration for connecting to AgentBus via gRPC.

    Attributes:
        bus_id: The AgentBus ID to use for operations
        host: Hostname or IP address of the AgentBus server
        port: Port number of the AgentBus server
    """

    bus_id: str
    host: str
    port: int
