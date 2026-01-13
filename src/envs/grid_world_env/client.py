from core.http_env_client import HTTPEnvClient
from core.client_types import StepResult
from .models import GridWorldAction, GridWorldObservation, GridWorldState, MoveAction


class GridWorldEnv(HTTPEnvClient[GridWorldAction, GridWorldObservation]):
    """
    A type-safe client for interacting with the GridWorld environment.
    This client inherits from HTTPEnvClient and is configured with the
    GridWorld dataclasses for automatic (de)serialization.
    
    Example:
        >>> env = GridWorldEnv.from_docker_image("grid-world-env:latest")
        >>> result = env.reset()
        >>> print(result.observation.message)
        >>> result = env.step_move(MoveAction.RIGHT)
        >>> print(f"Pos: [{result.observation.x}, {result.observation.y}]")
        >>> env.close()
    """

    # Added this __init__ method ===
    # This tells the base client which model to use for the .state() method.
    def __init__(self, *args, **kwargs):
        super().__init__(
            action_model=GridWorldAction,
            observation_model=GridWorldObservation,
            state_model=GridWorldState, 
            *args, 
            **kwargs
        )
    # ==========================================

    def step_move(self, move: MoveAction) -> StepResult[GridWorldObservation]:
        """
        Helper method to send a simple move action.
        
        Args:
            move: The MoveAction enum (e.g., MoveAction.UP)
        """
        action_payload = GridWorldAction(action=move)
        # 'super().step' comes from the base HTTPEnvClient
        return super().step(action_payload)