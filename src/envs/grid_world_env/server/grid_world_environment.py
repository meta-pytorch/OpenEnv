import uuid
from core.env_server import Environment
from ..models import (
    GridWorldAction, 
    GridWorldObservation, 
    GridWorldState, 
    MoveAction
)

class GridWorldEnvironment(Environment): 
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.grid_size = 5
        self.goal_pos = [4, 4]
        
        # Initialize State
        self._state = GridWorldState(
            agent_x=0,
            agent_y=0,
            goal_x=self.goal_pos[0],
            goal_y=self.goal_pos[1],
            grid_size=self.grid_size,
            episode_steps=0,
            step_count=0  # Initialize the standard counter
        )

    def reset(self) -> GridWorldObservation:
        print("Resetting Grid World environment...")
        
        # Update State
        self._state.agent_x = 0
        self._state.agent_y = 0
        self._state.episode_steps = 0
        
        # === FIX 1: Standard OpenEnv State tracking ===
        self._state.step_count = 0
        self._state.episode_id = str(uuid.uuid4())
        # ==============================================
        
        return GridWorldObservation(
            x=self._state.agent_x,
            y=self._state.agent_y,
            message="Welcome to Grid World! Goal is at [4, 4].",
            reward=None,
            done=False
        )

    def step(self, action: GridWorldAction) -> GridWorldObservation:
        self._state.step_count += 1
        # =============================================
        self._state.episode_steps += 1
        
        # Use current state
        current_x = self._state.agent_x
        current_y = self._state.agent_y
        
        move = action.action

        if move == MoveAction.UP:
            current_x -= 1
        elif move == MoveAction.DOWN:
            current_x += 1
        elif move == MoveAction.LEFT:
            current_y -= 1
        elif move == MoveAction.RIGHT:
            current_y += 1

        # Clamp to boundaries
        current_x = max(0, min(current_x, self.grid_size - 1))
        current_y = max(0, min(current_y, self.grid_size - 1))

        # Update State
        self._state.agent_x = current_x
        self._state.agent_y = current_y

        # Logic
        done = False
        message = "Keep going..."
        reward = -0.1

        if [current_x, current_y] == self.goal_pos:
            reward = 1.0
            done = True
            message = "You found the goal!"
        
        return GridWorldObservation(
            x=current_x,
            y=current_y,
            message=message,
            reward=reward,
            done=done
        )

    @property
    def state(self) -> GridWorldState:
        return self._state
