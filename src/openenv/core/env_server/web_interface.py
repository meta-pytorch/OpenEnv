# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Web interface for OpenEnv environments.

This module provides a web-based interface for interacting with OpenEnv environments,
including a two-pane layout for HumanAgent interaction and state observation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict

from .interfaces import Environment
from .serialization import deserialize_action_with_preprocessing, serialize_observation
from .types import Action, Observation, State, EnvironmentMetadata


def load_environment_metadata(
    env: Environment, env_name: Optional[str] = None
) -> EnvironmentMetadata:
    """
    Load environment metadata including README content.

    Args:
        env: The environment instance
        env_name: Optional environment name for README file lookup

    Returns:
        EnvironmentMetadata with loaded information
    """
    # Try to get metadata from environment if it has a method for it
    if hasattr(env, "get_metadata"):
        return env.get_metadata()

    # Default metadata
    metadata = EnvironmentMetadata(
        name=env_name or env.__class__.__name__,
        description=f"{env.__class__.__name__} environment",
        version="1.0.0",
    )

    # Try to load README from file system
    readme_content = _load_readme_from_filesystem(env_name)
    if readme_content:
        metadata.readme_content = readme_content

    return metadata


def _load_readme_from_filesystem(env_name: Optional[str]) -> Optional[str]:
    """
    Load README content from the filesystem.

    Tries multiple locations:
    1. Container filesystem: /app/README.md
    2. Local development: src/envs/{env_name}/README.md
    3. Environment variable: ENV_README_PATH
    """
    import os
    from pathlib import Path

    # Try container filesystem first
    container_readme = Path("/app/README.md")
    if container_readme.exists():
        try:
            return container_readme.read_text(encoding="utf-8")
        except Exception:
            pass

    # Try environment variable path
    custom_path = os.environ.get("ENV_README_PATH")
    if custom_path and Path(custom_path).exists():
        try:
            return Path(custom_path).read_text(encoding="utf-8")
        except Exception:
            pass

    # Try local development path
    if env_name:
        local_readme = Path(f"src/envs/{env_name}/README.md")
        if local_readme.exists():
            try:
                return local_readme.read_text(encoding="utf-8")
            except Exception:
                pass

    return None


class ActionLog(BaseModel):
    """Log entry for an action taken."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    timestamp: str = Field(description="Timestamp when action was taken")
    action: Dict[str, Any] = Field(description="Action that was taken")
    observation: Dict[str, Any] = Field(description="Observation returned from action")
    reward: Optional[float] = Field(
        default=None, description="Reward received from action"
    )
    done: bool = Field(description="Whether the episode is done after this action")
    step_count: int = Field(description="Step count when this action was taken")


class EpisodeState(BaseModel):
    """Current episode state for the web interface."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: Optional[str] = Field(default=None, description="Current episode ID")
    step_count: int = Field(description="Current step count in episode")
    current_observation: Optional[Dict[str, Any]] = Field(
        default=None, description="Current observation"
    )
    action_logs: List[ActionLog] = Field(
        default_factory=list, description="List of action logs"
    )
    is_reset: bool = Field(
        default=True, description="Whether the episode has been reset"
    )


class WebInterfaceManager:
    """Manages the web interface for an environment."""

    def __init__(
        self,
        env: Environment,
        action_cls: Type[Action],
        observation_cls: Type[Observation],
        metadata: Optional[EnvironmentMetadata] = None,
    ):
        self.env = env
        self.action_cls = action_cls
        self.observation_cls = observation_cls
        self.metadata = metadata or EnvironmentMetadata(
            name=env.__class__.__name__,
            description=f"{env.__class__.__name__} environment",
        )
        self.episode_state = EpisodeState(
            episode_id=None, step_count=0, current_observation=None, action_logs=[]
        )
        self.connected_clients: List[WebSocket] = []

    async def connect_websocket(self, websocket: WebSocket):
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.connected_clients.append(websocket)

        # Send current state to the new client
        await self._send_state_update()

    async def disconnect_websocket(self, websocket: WebSocket):
        """Disconnect a WebSocket client."""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)

    async def _send_state_update(self):
        """Send current state to all connected clients."""
        if not self.connected_clients:
            return

        state_data = {
            "type": "state_update",
            "episode_state": self.episode_state.model_dump(),
        }

        # Send to all connected clients
        disconnected_clients = []
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(state_data))
            except Exception:
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.remove(client)

    async def reset_environment(self) -> Dict[str, Any]:
        """Reset the environment and update state."""
        observation: Observation = self.env.reset()
        state: State = self.env.state

        # Serialize observation once using shared utility
        serialized = serialize_observation(observation)

        # Update episode state
        self.episode_state.episode_id = state.episode_id
        self.episode_state.step_count = 0
        self.episode_state.current_observation = serialized["observation"]
        self.episode_state.action_logs = []
        self.episode_state.is_reset = True

        # Send state update
        await self._send_state_update()

        return serialized

    async def step_environment(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step in the environment and update state."""
        # Deserialize action with preprocessing for web interface special cases
        action: Action = deserialize_action_with_preprocessing(
            action_data, self.action_cls
        )

        # Execute step
        observation: Observation = self.env.step(action)
        state: State = self.env.state

        # Serialize observation once using shared utility
        serialized = serialize_observation(observation)

        # Create action log
        action_log = ActionLog(
            timestamp=datetime.now().isoformat(),
            action=action.model_dump(exclude={"metadata"}),
            observation=serialized["observation"],
            reward=observation.reward,
            done=observation.done,
            step_count=state.step_count,
        )

        # Update episode state
        self.episode_state.episode_id = state.episode_id
        self.episode_state.step_count = state.step_count
        self.episode_state.current_observation = serialized["observation"]
        self.episode_state.action_logs.append(action_log)
        self.episode_state.is_reset = False

        # Send state update
        await self._send_state_update()

        return serialized

    def get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        state: State = self.env.state
        return state.model_dump()


def create_web_interface_app(
    env: Environment,
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
) -> FastAPI:
    """
    Create a FastAPI application with web interface for the given environment.

    Args:
        env: The Environment instance to serve
        action_cls: The Action subclass this environment expects
        observation_cls: The Observation subclass this environment returns
        env_name: Optional environment name for README loading

    Returns:
        FastAPI application instance with web interface
    """
    from .http_server import create_fastapi_app

    # Create the base environment app
    app = create_fastapi_app(env, action_cls, observation_cls)

    # Load environment metadata
    metadata = load_environment_metadata(env, env_name)

    # Create web interface manager
    web_manager = WebInterfaceManager(env, action_cls, observation_cls, metadata)

    # Add web interface routes
    @app.get("/web", response_class=HTMLResponse)
    async def web_interface():
        """Serve the web interface."""
        return get_web_interface_html(action_cls, web_manager.metadata)

    @app.get("/web/metadata")
    async def web_metadata():
        """Get environment metadata."""
        return web_manager.metadata.model_dump()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)

    @app.post("/web/reset")
    async def web_reset():
        """Reset endpoint for web interface."""
        return await web_manager.reset_environment()

    @app.post("/web/step")
    async def web_step(request: Dict[str, Any]):
        """Step endpoint for web interface."""
        # Check if this is a message-based request (chat environment)
        if "message" in request:
            message = request["message"]
            # Convert message to action using the environment's message_to_action method
            action = web_manager.env.message_to_action(message)
            action_data = {"tokens": action.tokens.tolist()}
        else:
            action_data = request.get("action", {})

        return await web_manager.step_environment(action_data)

    @app.get("/web/state")
    async def web_state():
        """State endpoint for web interface."""
        return web_manager.get_state()

    return app


def get_web_interface_html(
    action_cls: Type[Action], metadata: Optional[EnvironmentMetadata] = None
) -> str:
    """Generate the HTML for the web interface."""

    # Check if this is a chat environment by looking for tokens field
    is_chat_env = False
    if hasattr(action_cls, "model_fields"):
        for field_name, field_info in action_cls.model_fields.items():
            if (
                field_name == "tokens"
                and hasattr(field_info.annotation, "__name__")
                and "Tensor" in field_info.annotation.__name__
            ):
                is_chat_env = True
                break

    # Get action fields for dynamic form generation with enhanced metadata
    action_fields = _extract_action_fields(action_cls)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenEnv Web Interface</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            height: 100vh;
            overflow: hidden;
        }}
        
        .container {{
            display: flex;
            height: 100vh;
        }}
        
        .left-pane {{
            width: 50%;
            background: white;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
        }}
        
        .right-pane {{
            width: 50%;
            background: #fafafa;
            display: flex;
            flex-direction: column;
        }}
        
        .pane-header {{
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
            background: #f8f9fa;
            font-weight: 600;
            font-size: 16px;
        }}
        
        .pane-content {{
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }}
        
        .action-form {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .form-group {{
            margin-bottom: 15px;
        }}
        
        .form-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #333;
        }}
        
        .form-group input, .form-group textarea {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .form-group input:focus, .form-group textarea:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }}
        
        .btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
            margin-bottom: 10px;
        }}
        
        .btn:hover {{
            background: #0056b3;
        }}
        
        .btn:disabled {{
            background: #6c757d;
            cursor: not-allowed;
        }}
        
        .btn-secondary {{
            background: #6c757d;
        }}
        
        .btn-secondary:hover {{
            background: #545b62;
        }}
        
        .state-display {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        
        .state-item {{
            margin-bottom: 8px;
        }}
        
        .state-label {{
            font-weight: 500;
            color: #666;
        }}
        
        .state-value {{
            color: #333;
            font-family: monospace;
        }}
        
        .logs-container {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .log-entry {{
            border-bottom: 1px solid #f0f0f0;
            padding: 10px 0;
        }}
        
        .log-entry:last-child {{
            border-bottom: none;
        }}
        
        .log-timestamp {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .log-action {{
            background: #e3f2fd;
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 5px;
            font-family: monospace;
            font-size: 12px;
        }}
        
        .log-observation {{
            background: #f3e5f5;
            padding: 8px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
        }}
        
        .log-reward {{
            font-weight: 600;
            color: #28a745;
        }}
        
        .log-done {{
            font-weight: 600;
            color: #dc3545;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-connected {{
            background: #28a745;
        }}
        
        .status-disconnected {{
            background: #dc3545;
        }}
        
        .json-display {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 10px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }}
        
        /* Chat Interface Styles */
        .chat-interface {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .chat-messages {{
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .chat-message {{
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
        }}
        
        .chat-message:last-child {{
            margin-bottom: 0;
        }}
        
        .chat-message.user {{
            background: #e3f2fd;
            margin-left: 20px;
        }}
        
        .chat-message.assistant {{
            background: #f3e5f5;
            margin-right: 20px;
        }}
        
        .chat-message.system {{
            background: #e8f5e8;
            font-style: italic;
        }}
        
        .message-role {{
            font-weight: 600;
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .message-content {{
            font-size: 14px;
            line-height: 1.4;
        }}
        
        .chat-input-container {{
            border-top: 1px solid #e0e0e0;
            padding-top: 15px;
        }}
        
        .role-selector {{
            margin-bottom: 10px;
        }}
        
        .role-selector label {{
            font-weight: 500;
            margin-right: 10px;
        }}
        
        .role-selector select {{
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        
        .message-input {{
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }}
        
        .message-input textarea {{
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
            font-family: inherit;
        }}
        
        .message-input textarea:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }}
        
        /* Instructions Section Styles */
        .instructions-section {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .instructions-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .instructions-title {{
            font-size: 18px;
            font-weight: 600;
            color: #333;
            margin: 0;
        }}
        
        .instructions-toggle {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 5px 10px;
            cursor: pointer;
            font-size: 12px;
            color: #6c757d;
        }}
        
        .instructions-toggle:hover {{
            background: #e9ecef;
        }}
        
        .instructions-content {{
            display: none;
            max-height: 400px;
            overflow-y: auto;
            border-top: 1px solid #e0e0e0;
            padding-top: 15px;
        }}
        
        .instructions-content.expanded {{
            display: block;
        }}
        
        .instructions-content h1,
        .instructions-content h2,
        .instructions-content h3 {{
            color: #333;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        
        .instructions-content h1 {{
            font-size: 24px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        
        .instructions-content h2 {{
            font-size: 20px;
        }}
        
        .instructions-content h3 {{
            font-size: 16px;
        }}
        
        .instructions-content p {{
            margin-bottom: 10px;
            line-height: 1.6;
        }}
        
        .instructions-content code {{
            background: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 14px;
        }}
        
        .instructions-content pre {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            margin: 10px 0;
        }}
        
        .instructions-content pre code {{
            background: none;
            padding: 0;
        }}
        
        .instructions-content ul,
        .instructions-content ol {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        
        .instructions-content li {{
            margin-bottom: 5px;
        }}
        
        .instructions-content table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        
        .instructions-content th,
        .instructions-content td {{
            border: 1px solid #dee2e6;
            padding: 8px 12px;
            text-align: left;
        }}
        
        .instructions-content th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        
        /* Enhanced Form Styles */
        .help-text {{
            display: block;
            margin-top: 5px;
            font-size: 12px;
            color: #6c757d;
            font-style: italic;
        }}
        
        .form-group label {{
            font-weight: 500;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .form-group select {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            background-color: white;
        }}
        
        .form-group select:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }}
        
        .form-group textarea {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
        }}
        
        .form-group textarea:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }}
        
        .form-group input[type="number"] {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .form-group input[type="number"]:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }}
        
        .form-group input[type="text"]:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }}
        
        .required-indicator {{
            color: #dc3545;
            font-weight: bold;
        }}
        
        .form-group .field-description {{
            font-size: 11px;
            color: #666;
            margin-top: 2px;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Left Pane: HumanAgent Interface -->
        <div class="left-pane">
            <div class="pane-header">
                <span class="status-indicator status-disconnected" id="connection-status"></span>
                HumanAgent Interface
            </div>
            <div class="pane-content">
                <!-- Instructions Section -->
                {_generate_instructions_section(metadata)}
                
                <!-- Action Form or Chat Interface -->
                {_generate_action_interface(action_fields, is_chat_env)}
                
                <!-- Control Buttons -->
                <div style="margin-bottom: 20px;">
                    <button class="btn btn-secondary" id="reset-btn">Reset Environment</button>
                    <button class="btn btn-secondary" id="state-btn">Get State</button>
                </div>
                
                <!-- Current State Display -->
                <div class="state-display">
                    <h3>Current State</h3>
                    <div id="current-state">
                        <div class="state-item">
                            <span class="state-label">Status:</span>
                            <span class="state-value" id="env-status">Not initialized</span>
                        </div>
                        <div class="state-item">
                            <span class="state-label">Episode ID:</span>
                            <span class="state-value" id="episode-id">-</span>
                        </div>
                        <div class="state-item">
                            <span class="state-label">Step Count:</span>
                            <span class="state-value" id="step-count">0</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Right Pane: State Observer -->
        <div class="right-pane">
            <div class="pane-header">
                State Observer
            </div>
            <div class="pane-content">
                <!-- Current Observation -->
                <div class="state-display">
                    <h3>Current Observation</h3>
                    <div id="current-observation" class="json-display">
                        No observation yet
                    </div>
                </div>
                
                <!-- Action Logs -->
                <div class="logs-container">
                    <h3>Action History</h3>
                    <div id="action-logs">
                        No actions taken yet
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        class OpenEnvWebInterface {{
            constructor() {{
                this.ws = null;
                this.isConnected = false;
                this.init();
            }}
            
            init() {{
                this.connectWebSocket();
                this.setupEventListeners();
            }}
            
            connectWebSocket() {{
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${{protocol}}//${{window.location.host}}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {{
                    this.isConnected = true;
                    this.updateConnectionStatus(true);
                    console.log('WebSocket connected');
                }};
                
                this.ws.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    if (data.type === 'state_update') {{
                        this.updateUI(data.episode_state);
                    }}
                }};
                
                this.ws.onclose = () => {{
                    this.isConnected = false;
                    this.updateConnectionStatus(false);
                    console.log('WebSocket disconnected');
                    // Attempt to reconnect after 3 seconds
                    setTimeout(() => this.connectWebSocket(), 3000);
                }};
                
                this.ws.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                }};
            }}
            
            setupEventListeners() {{
                // Instructions toggle
                const instructionsToggle = document.getElementById('instructions-toggle');
                const instructionsContent = document.getElementById('instructions-content');
                if (instructionsToggle && instructionsContent) {{
                    instructionsToggle.addEventListener('click', () => {{
                        instructionsContent.classList.toggle('expanded');
                        instructionsToggle.textContent = instructionsContent.classList.contains('expanded') 
                            ? 'Hide Instructions' : 'Show Instructions';
                    }});
                }}
                
                // Check if this is a chat environment
                const isChatEnv = document.getElementById('chat-messages') !== null;
                
                if (isChatEnv) {{
                    // Chat environment event listeners
                    document.getElementById('send-message-btn').addEventListener('click', () => {{
                        this.sendMessage();
                    }});
                    
                    // Send message on Enter (but allow Shift+Enter for new lines)
                    document.getElementById('message-input').addEventListener('keydown', (e) => {{
                        if (e.key === 'Enter' && !e.shiftKey) {{
                            e.preventDefault();
                            this.sendMessage();
                        }}
                    }});
                }} else {{
                    // Traditional action form submission
                    const actionForm = document.getElementById('action-form');
                    if (actionForm) {{
                        actionForm.addEventListener('submit', (e) => {{
                            e.preventDefault();
                            this.submitAction();
                        }});
                    }}
                }}
                
                // Reset button
                document.getElementById('reset-btn').addEventListener('click', () => {{
                    this.resetEnvironment();
                }});
                
                // State button
                document.getElementById('state-btn').addEventListener('click', () => {{
                    this.getState();
                }});
            }}
            
            async sendMessage() {{
                const messageInput = document.getElementById('message-input');
                const roleSelect = document.getElementById('message-role');
                const message = messageInput.value.trim();
                const role = roleSelect.value;
                
                if (!message) {{
                    return;
                }}
                
                // Add message to chat display immediately
                this.addMessageToChat(role, message);
                
                // Clear input
                messageInput.value = '';
                
                try {{
                    // Send message to server to convert to action and step
                    const response = await fetch('/web/step', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ 
                            message: {{
                                role: role,
                                content: message
                            }}
                        }})
                    }});
                    
                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}
                    
                    const result = await response.json();
                    console.log('Message sent:', result);
                }} catch (error) {{
                    console.error('Error sending message:', error);
                    alert('Error sending message: ' + error.message);
                }}
            }}
            
            addMessageToChat(role, content) {{
                const chatMessages = document.getElementById('chat-messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `chat-message ${{role}}`;
                
                messageDiv.innerHTML = `
                    <div class="message-role">${{role.charAt(0).toUpperCase() + role.slice(1)}}</div>
                    <div class="message-content">${{content}}</div>
                `;
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }}
            
            async submitAction() {{
                const formData = new FormData(document.getElementById('action-form'));
                const action = {{}};
                
                // Collect form data
                for (const [key, value] of formData.entries()) {{
                    if (value !== '') {{
                        // Handle tensor fields (tokens) - convert comma-separated string to array
                        if (key === 'tokens') {{
                            try {{
                                action[key] = value.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));
                            }} catch (e) {{
                                console.error('Error parsing tokens:', e);
                                action[key] = [];
                            }}
                        }} else {{
                            action[key] = value;
                        }}
                    }}
                }}
                
                try {{
                    const response = await fetch('/web/step', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ action }})
                    }});
                    
                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}
                    
                    const result = await response.json();
                    console.log('Step result:', result);
                }} catch (error) {{
                    console.error('Error submitting action:', error);
                    alert('Error submitting action: ' + error.message);
                }}
            }}
            
            async resetEnvironment() {{
                try {{
                    const response = await fetch('/web/reset', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }}
                    }});
                    
                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}
                    
                    const result = await response.json();
                    console.log('Reset result:', result);
                }} catch (error) {{
                    console.error('Error resetting environment:', error);
                    alert('Error resetting environment: ' + error.message);
                }}
            }}
            
            async getState() {{
                try {{
                    const response = await fetch('/web/state');
                    const state = await response.json();
                    console.log('Current state:', state);
                    alert('Current state: ' + JSON.stringify(state, null, 2));
                }} catch (error) {{
                    console.error('Error getting state:', error);
                    alert('Error getting state: ' + error.message);
                }}
            }}
            
            updateConnectionStatus(connected) {{
                const indicator = document.getElementById('connection-status');
                if (connected) {{
                    indicator.className = 'status-indicator status-connected';
                }} else {{
                    indicator.className = 'status-indicator status-disconnected';
                }}
            }}
            
            updateUI(episodeState) {{
                // Check if this is a chat environment
                const isChatEnv = document.getElementById('chat-messages') !== null;
                
                // Update current state
                document.getElementById('env-status').textContent = 
                    episodeState.is_reset ? 'Reset' : 'Running';
                document.getElementById('episode-id').textContent = 
                    episodeState.episode_id || '-';
                document.getElementById('step-count').textContent = 
                    episodeState.step_count.toString();
                
                if (isChatEnv) {{
                    // Update chat interface
                    this.updateChatInterface(episodeState);
                }} else {{
                    // Update traditional observation display
                    const observationDiv = document.getElementById('current-observation');
                    if (episodeState.current_observation) {{
                        observationDiv.textContent = JSON.stringify(
                            episodeState.current_observation, null, 2
                        );
                    }} else {{
                        observationDiv.textContent = 'No observation yet';
                    }}
                }}
                
                // Update action logs
                const logsDiv = document.getElementById('action-logs');
                if (episodeState.action_logs.length === 0) {{
                    logsDiv.innerHTML = 'No actions taken yet';
                }} else {{
                    logsDiv.innerHTML = episodeState.action_logs.map(log => `
                        <div class="log-entry">
                            <div class="log-timestamp">${{log.timestamp}} (Step ${{log.step_count}})</div>
                            <div class="log-action">Action: ${{JSON.stringify(log.action, null, 2)}}</div>
                            <div class="log-observation">Observation: ${{JSON.stringify(log.observation, null, 2)}}</div>
                            <div>
                                <span class="log-reward">Reward: ${{log.reward !== null ? log.reward : 'None'}}</span>
                                ${{log.done ? '<span class="log-done">DONE</span>' : ''}}
                            </div>
                        </div>
                    `).join('');
                }}
            }}
            
            updateChatInterface(episodeState) {{
                const chatMessages = document.getElementById('chat-messages');
                if (!chatMessages) return;
                
                // Clear existing messages (except system message)
                const systemMessage = chatMessages.querySelector('.chat-message.system');
                chatMessages.innerHTML = '';
                if (systemMessage) {{
                    chatMessages.appendChild(systemMessage);
                }}
                
                // Add messages from current observation
                if (episodeState.current_observation && episodeState.current_observation.messages) {{
                    episodeState.current_observation.messages.forEach(msg => {{
                        this.addMessageToChat(msg.role, msg.content);
                    }});
                }}
            }}
        }}
        
        // Initialize the web interface when the page loads
        document.addEventListener('DOMContentLoaded', () => {{
            new OpenEnvWebInterface();
        }});
    </script>
</body>
</html>
    """.replace(
        "{_generate_action_form_fields(action_fields)}",
        _generate_action_form_fields(action_fields),
    )


def _generate_instructions_section(metadata: Optional[EnvironmentMetadata]) -> str:
    """Generate the instructions section with environment documentation."""
    if not metadata or not metadata.readme_content:
        return ""

    html_content = _markdown_to_html(metadata.readme_content)

    return f"""
                <!-- Instructions Section -->
                <div class="instructions-section">
                    <div class="instructions-header">
                        <h3 class="instructions-title">{metadata.name}</h3>
                        <button class="instructions-toggle" id="instructions-toggle">Show Instructions</button>
                    </div>
                    <div class="instructions-content" id="instructions-content">
                        <div class="instructions-readme">
                            {html_content}
                        </div>
                    </div>
                </div>
    """


def _extract_action_fields(action_cls: Type[Action]) -> List[Dict[str, Any]]:
    """Extract enhanced field metadata from Action class for form generation."""
    # Use Pydantic's JSON schema generation for robust metadata extraction
    try:
        schema = action_cls.model_json_schema()
    except AttributeError:
        # Fallback for non-Pydantic v2 models or if something goes wrong
        return []

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    action_fields = []

    for field_name, field_info in properties.items():
        if field_name == "metadata":
            continue

        # JSON schema "type" can be a string or list/undefined
        # Determine our internal input type
        input_type = _determine_input_type_from_schema(field_info, field_name)

        is_required = field_name in required_fields

        action_fields.append(
            {
                "name": field_name,
                "type": input_type,
                "required": is_required,
                "description": field_info.get("description", ""),
                "default_value": field_info.get("default"),
                "choices": field_info.get("enum"),
                "min_value": field_info.get("minimum"),
                "max_value": field_info.get("maximum"),
                "min_length": field_info.get("minLength"),
                "max_length": field_info.get("maxLength"),
                "pattern": field_info.get("pattern"),
                "placeholder": _generate_placeholder(field_name, field_info),
                "help_text": _generate_help_text(field_name, field_info),
            }
        )

    return action_fields


def _determine_input_type_from_schema(
    field_info: Dict[str, Any], field_name: str
) -> str:
    """Determine the appropriate HTML input type from JSON schema info."""
    schema_type = field_info.get("type")

    # Check for specific tensor field convention
    if "tokens" in field_name.lower():
        return "tensor"

    if "enum" in field_info:
        return "select"

    if schema_type == "boolean":
        return "checkbox"

    if schema_type == "integer" or schema_type == "number":
        return "number"

    if schema_type == "string":
        # Check if it should be a textarea
        if (
            field_info.get("maxLength", 0) > 100
            or "message" in field_name.lower()
            or "code" in field_name.lower()
        ):
            return "textarea"
        return "text"

    # Default fallback
    return "text"


def _generate_placeholder(field_name: str, field_info: Dict[str, Any]) -> str:
    """Generate placeholder text."""
    if "message" in field_name.lower():
        return f"Enter {field_name.replace('_', ' ')}..."
    elif "code" in field_name.lower():
        return "Enter Python code here..."
    elif "tokens" in field_name.lower():
        return "Enter comma-separated token IDs (e.g., 1,2,3,4,5)"
    else:
        return f"Enter {field_name.replace('_', ' ')}..."


def _generate_help_text(field_name: str, field_info: Dict[str, Any]) -> str:
    """Generate help text."""
    description = field_info.get("description", "")
    if description:
        return description

    if "action_id" in field_name.lower():
        return "The action ID to execute in environment"
    elif "game_name" in field_name.lower():
        return "Name of game or environment"
    elif "tokens" in field_name.lower():
        return "Token IDs as a comma-separated list of integers"
    elif "code" in field_name.lower():
        return "Python code to execute in environment"
    elif "message" in field_name.lower():
        return "Text message to send"

    return ""


def _markdown_to_html(markdown: str) -> str:
    """Convert basic markdown to HTML for README display."""
    import html
    import re

    # Escape HTML first
    html_content = html.escape(markdown)

    # Convert headers
    html_content = re.sub(
        r"^# (.*?)$", r"<h1>\1</h1>", html_content, flags=re.MULTILINE
    )
    html_content = re.sub(
        r"^## (.*?)$", r"<h2>\1</h2>", html_content, flags=re.MULTILINE
    )
    html_content = re.sub(
        r"^### (.*?)$", r"<h3>\1</h3>", html_content, flags=re.MULTILINE
    )

    # Convert code blocks
    html_content = re.sub(
        r"```(.*?)\n(.*?)\n```",
        r"<pre><code>\2</code></pre>",
        html_content,
        flags=re.DOTALL,
    )
    html_content = re.sub(r"`([^`]+)`", r"<code>\1</code>", html_content)

    # Convert bold and italic
    html_content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html_content)
    html_content = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html_content)

    # Convert lists
    html_content = re.sub(
        r"^- (.*?)$", r"<li>\1</li>", html_content, flags=re.MULTILINE
    )
    html_content = re.sub(
        r"(<li>.*</li>)", r"<ul>\1</ul>", html_content, flags=re.DOTALL
    )

    # Convert line breaks
    html_content = html_content.replace("\n", "<br>")

    return html_content


def _generate_action_interface(
    action_fields: List[Dict[str, Any]], is_chat_env: bool
) -> str:
    """Generate either a chat interface or action form based on environment type."""
    if is_chat_env:
        return _generate_chat_interface()
    else:
        return _generate_action_form(action_fields)


def _generate_chat_interface() -> str:
    """Generate a chat-style interface for chat environments."""
    return """
                <!-- Chat Interface -->
                <div class="chat-interface">
                    <h3>Chat Interface</h3>
                    <div class="chat-messages" id="chat-messages">
                        <div class="chat-message system">
                            <div class="message-role">System</div>
                            <div class="message-content">Chat environment ready. Send a message to start the conversation.</div>
                        </div>
                    </div>
                    <div class="chat-input-container">
                        <div class="role-selector">
                            <label for="message-role">Role:</label>
                            <select id="message-role">
                                <option value="user">User</option>
                                <option value="assistant">Assistant</option>
                            </select>
                        </div>
                        <div class="message-input">
                            <textarea id="message-input" placeholder="Type your message here..." rows="3"></textarea>
                            <button class="btn" id="send-message-btn">Send Message</button>
                        </div>
                    </div>
                </div>
    """


def _generate_action_form(action_fields: List[Dict[str, Any]]) -> str:
    """Generate a traditional action form for non-chat environments."""
    return f"""
                <!-- Action Form -->
                <div class="action-form">
                    <h3>Take Action</h3>
                    <form id="action-form">
                        {_generate_action_form_fields(action_fields)}
                        <button type="submit" class="btn" id="step-btn">Step</button>
                    </form>
                </div>
    """


def _generate_action_form_fields(action_fields: List[Dict[str, Any]]) -> str:
    """Generate HTML form fields for action input with enhanced metadata."""
    if not action_fields:
        return "<p>No action fields available</p>"

    fields_html = []
    for field in action_fields:
        field_html = _generate_single_field(field)
        fields_html.append(field_html)

    return "\n".join(fields_html)


def _generate_single_field(field: Dict[str, Any]) -> str:
    """Generate HTML for a single form field with enhanced metadata."""
    field_name = field["name"]
    field_type = field["type"]
    required = field["required"]
    placeholder = field.get("placeholder", "")
    help_text = field.get("help_text", "")
    choices = field.get("choices", [])
    min_value = field.get("min_value")
    max_value = field.get("max_value")
    default_value = field.get("default_value")
    min_length = field.get("min_length")
    max_length = field.get("max_length")
    pattern = field.get("pattern")

    # Build label with required indicator
    label_text = field_name.replace("_", " ").title()
    if required:
        label_text += ' <span style="color: red;">*</span>'

    # Build input attributes
    input_attrs = []
    if required:
        input_attrs.append("required")
    if placeholder:
        input_attrs.append(f'placeholder="{placeholder}"')
    if min_value is not None:
        input_attrs.append(f'min="{min_value}"')
    if max_value is not None:
        input_attrs.append(f'max="{max_value}"')
    if min_length is not None:
        input_attrs.append(f'minlength="{min_length}"')
    if max_length is not None:
        input_attrs.append(f'maxlength="{max_length}"')
    if pattern is not None:
        input_attrs.append(f'pattern="{pattern}"')
    if default_value is not None:
        input_attrs.append(f'value="{default_value}"')

    attrs_str = " ".join(input_attrs)

    if field_type == "checkbox":
        checked = "checked" if default_value is True else ""
        return f'''
            <div class="form-group">
                <label>
                    <input type="checkbox" name="{field_name}" value="true" {checked}>
                    {label_text}
                </label>
                {f'<small class="help-text">{help_text}</small>' if help_text else ""}
            </div>
        '''

    elif field_type == "select":
        options_html = []
        if not required:
            options_html.append(f'<option value="">-- Select {label_text} --</option>')

        for choice in choices:
            selected = "selected" if str(choice) == str(default_value) else ""
            options_html.append(
                f'<option value="{choice}" {selected}>{choice}</option>'
            )

        return f'''
            <div class="form-group">
                <label for="{field_name}">{label_text}:</label>
                <select name="{field_name}" id="{field_name}" {attrs_str}>
                    {"".join(options_html)}
                </select>
                {f'<small class="help-text">{help_text}</small>' if help_text else ""}
            </div>
        '''

    elif field_type == "tensor":
        return f'''
            <div class="form-group">
                <label for="{field_name}">{label_text} (comma-separated integers):</label>
                <input type="text" name="{field_name}" id="{field_name}" {attrs_str}>
                <small class="help-text">{help_text or "Enter token IDs as comma-separated integers (e.g., 1,2,3,4,5)"}</small>
            </div>
        '''

    elif field_type == "textarea":
        return f'''
            <div class="form-group">
                <label for="{field_name}">{label_text}:</label>
                <textarea name="{field_name}" id="{field_name}" rows="3" {attrs_str}>{default_value if default_value is not None else ""}</textarea>
                {f'<small class="help-text">{help_text}</small>' if help_text else ""}
            </div>
        '''

    else:
        return f'''
            <div class="form-group">
                <label for="{field_name}">{label_text}:</label>
                <input type="{field_type}" name="{field_name}" id="{field_name}" {attrs_str}>
                {f'<small class="help-text">{help_text}</small>' if help_text else ""}
            </div>
        '''
