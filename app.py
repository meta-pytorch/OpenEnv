"""
Oversight Inbox Arena — HuggingFace Space Entry Point

This file is the main entry point for the HuggingFace Space.
It launches a pure Gradio app that also exposes the REST API
on /api/* via a sub-application mount.
"""

import sys
import os

# Ensure the environment packages are importable
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "envs"))
sys.path.insert(0, _root)

import gradio as gr

try:
    from envs.email_triage_env.server.ui import build_ui
except ImportError:
    try:
        from email_triage_env.server.ui import build_ui
    except ImportError:
        from server.ui import build_ui

# Build and launch the Gradio UI
demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
