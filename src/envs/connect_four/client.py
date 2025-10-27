from __future__ import annotations
import requests
from typing import Tuple
from .models import ConnectFourAction, ConnectFourObservation, ConnectFourState


class ConnectFourEnvClient:
    """
    Tiny HTTP client for the Connect Four server.

    Example:
        env = ConnectFourEnvClient("http://localhost:8020")
        obs, st = env.reset()
        obs, st = env.step(ConnectFourAction(column=3))
    """
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def reset(self) -> Tuple[ConnectFourObservation, ConnectFourState]:
        r = requests.post(f"{self.base}/reset", timeout=30)
        r.raise_for_status()
        payload = r.json()
        return ConnectFourObservation(**payload["observation"]), ConnectFourState(**payload["state"])

    def step(self, action: ConnectFourAction) -> Tuple[ConnectFourObservation, ConnectFourState]:
        r = requests.post(f"{self.base}/step", json=action.model_dump(), timeout=30)
        r.raise_for_status()
        payload = r.json()
        return ConnectFourObservation(**payload["observation"]), ConnectFourState(**payload["state"])

    def state(self) -> ConnectFourState:
        r = requests.get(f"{self.base}/state", timeout=15)
        r.raise_for_status()
        return ConnectFourState(**r.json())

    def close(self) -> None:
        try:
            requests.post(f"{self.base}/close", timeout=10)
        except Exception:
            pass
