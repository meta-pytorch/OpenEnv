# server/app.py
import os
from core.env_server import create_fastapi_app
from ..models import WildfireAction, WildfireObservation
from .wildfire_environment import WildfireEnvironment

W = int(os.getenv("WILDFIRE_WIDTH", "16"))
H = int(os.getenv("WILDFIRE_HEIGHT", "16"))
env = WildfireEnvironment(width=W, height=H)
app = create_fastapi_app(env, WildfireAction, WildfireObservation)
