from __future__ import annotations
import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from ..models import ConnectFourAction, ConnectFourObservation, ConnectFourState
from .connect_four_environment import (
    ConnectFourEnvironment,
    ConnectFourConfig,
)

# ------------ env config from environment variables ------------
PORT = int(os.getenv("PORT", "8020"))
GAME_STRING = os.getenv("OPENSPIEL_GAME", "connect_four")
AUTO_OPP = os.getenv("CONNECT4_AUTOPLAY_OPPONENT", "false").lower() in {"1", "true", "yes"}
OPP_POLICY = os.getenv("CONNECT4_OPP_POLICY", "random")  # random | lowest | highest

# ------------------------- FastAPI app -------------------------
app = FastAPI(title="OpenEnv • Connect Four (OpenSpiel)", version="1.0.0")

_env: Optional[ConnectFourEnvironment] = None
_state = ConnectFourState()

def _dump(model: BaseModel) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()

def _ensure_env() -> ConnectFourEnvironment:
    global _env
    if _env is None:
        cfg = ConnectFourConfig(
            game_string=GAME_STRING,
            autoplay_opponent=AUTO_OPP,
            opponent_policy=OPP_POLICY,
        )
        _env = ConnectFourEnvironment(cfg)
    return _env

# --------------------------- endpoints --------------------------

@app.post("/reset")
def reset():
    env = _ensure_env()
    obs_dict, st_dict = env.reset()
    global _state
    _state = ConnectFourState(**st_dict)
    return {"observation": _dump(ConnectFourObservation(**obs_dict)), "state": _dump(_state)}

@app.post("/step")
def step(action: ConnectFourAction):
    env = _ensure_env()
    obs_dict, st_dict = env.step(action.column)
    global _state
    _state = ConnectFourState(**st_dict)
    return {"observation": _dump(ConnectFourObservation(**obs_dict)), "state": _dump(_state)}

@app.get("/state")
def state():
    return _dump(_state)

@app.post("/close")
def close():
    global _env
    try:
        if _env is not None:
            _env.close()
    finally:
        _env = None
    return {"ok": True}
