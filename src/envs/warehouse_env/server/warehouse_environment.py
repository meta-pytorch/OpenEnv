"""
Core warehouse environment implementation.

This module implements the warehouse logistics optimization environment
with grid-based navigation, package pickup/delivery, and reward calculation.
"""

import random
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from core.client_types import StepResult

from core.env_server import Environment
from envs.warehouse_env.models import (
    Package,
    WarehouseAction,
    WarehouseObservation,
    WarehouseState,
)


# Cell types
EMPTY = 0
WALL = 1
SHELF = 2
PICKUP_ZONE = 3
DROPOFF_ZONE = 4


# Difficulty configurations
DIFFICULTY_CONFIGS = {
    1: {"grid_size": (5, 5), "num_packages": 1, "num_obstacles": 0, "max_steps": 50},
    2: {"grid_size": (8, 8), "num_packages": 2, "num_obstacles": 3, "max_steps": 100},
    3: {"grid_size": (10, 10), "num_packages": 3, "num_obstacles": 8, "max_steps": 150},
    4: {
        "grid_size": (15, 15),
        "num_packages": 5,
        "num_obstacles": 20,
        "max_steps": 250,
    },
    5: {
        "grid_size": (20, 20),
        "num_packages": 8,
        "num_obstacles": 40,
        "max_steps": 400,
    },
}


class WarehouseEnvironment(Environment):
    """
    Warehouse optimization environment.

    A grid-based environment where a robot must navigate a warehouse,
    pick up packages from pickup zones, and deliver them to dropoff zones
    while avoiding obstacles.
    """

    def __init__(
        self,
        difficulty_level: int = 2,
        grid_width: Optional[int] = None,
        grid_height: Optional[int] = None,
        num_packages: Optional[int] = None,
        max_steps: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the warehouse environment.

        Args:
            difficulty_level: Preset difficulty (1-5)
            grid_width: Custom grid width (overrides difficulty)
            grid_height: Custom grid height (overrides difficulty)
            num_packages: Custom package count (overrides difficulty)
            max_steps: Custom step limit (overrides difficulty)
            random_seed: Random seed for reproducibility
        """
        super().__init__()

        # Get config from difficulty or use custom values
        config = DIFFICULTY_CONFIGS.get(difficulty_level, DIFFICULTY_CONFIGS[2])

        self.difficulty_level = difficulty_level
        self.grid_width = grid_width or config["grid_size"][0]
        self.grid_height = grid_height or config["grid_size"][1]
        self.num_packages = num_packages or config["num_packages"]
        self.max_steps = max_steps or config["max_steps"]
        self.num_obstacles = config["num_obstacles"]

        if random_seed is not None:
            random.seed(random_seed)

        # Episode state
        self.episode_id: str = ""
        self.step_count: int = 0
        self.grid: List[List[int]] = []
        self.robot_position: Tuple[int, int] = (0, 0)
        self.robot_carrying: Optional[int] = None
        self.packages: List[Package] = []
        self.packages_delivered: int = 0
        self.cum_reward: float = 0.0
        self.is_done: bool = False

        # Pickup and dropoff zones
        self.pickup_zones: List[Tuple[int, int]] = []
        self.dropoff_zones: List[Tuple[int, int]] = []

    def reset(self) -> WarehouseObservation:
        """Reset the environment for a new episode."""
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.packages_delivered = 0
        self.cum_reward = 0.0
        self.is_done = False
        self.robot_carrying = None

        # Generate warehouse layout
        self._generate_warehouse()

        # Place robot at start position (usually near center)
        self.robot_position = (self.grid_width // 2, self.grid_height // 2)

        # Generate packages
        self._generate_packages()

        observation = self._get_observation(
            action_success=True,
            message="Warehouse environment ready! Navigate to pickup zones to collect packages.",
        )

        return observation

    def step(self, action: WarehouseAction) -> WarehouseObservation:
        """Execute an action and return the result."""
        if self.is_done:
            return self._get_observation(False, "Episode already finished")

        self.step_count += 1
        reward = 0.0
        action_success = False
        message = ""

        # Track state before action
        packages_delivered_before = self.packages_delivered

        # Execute action
        if action.action_id in [0, 1, 2, 3]:  # Movement actions
            action_success, message = self._move_robot(action.action_id)
            reward = -0.1  # Small step penalty
            if not action_success:
                reward = -1.0  # Penalty for invalid move

        elif action.action_id == 4:  # PICK_UP
            action_success, message = self._pickup_package()
            if action_success:
                reward = 10.0  # Reward for successful pickup
            else:
                reward = -1.0  # Penalty for invalid pickup

        elif action.action_id == 5:  # DROP_OFF
            action_success, message = self._dropoff_package()
            if action_success:
                # Major reward for delivery
                reward = 100.0

                # Time bonus
                time_bonus = (self.max_steps - self.step_count) * 0.1
                reward += time_bonus

                # Check if all packages delivered
                if self.packages_delivered == self.num_packages:
                    reward += 200.0  # Completion bonus
                    self.is_done = True
                    message += " All packages delivered! Episode complete!"
            else:
                reward = -1.0

        # Update package waiting times
        for package in self.packages:
            if package.status == "waiting":
                package.time_waiting += 1

        # Check timeout
        if self.step_count >= self.max_steps:
            self.is_done = True
            message += " Maximum steps reached. Episode terminated."

        self.cum_reward += reward

        observation = self._get_observation(action_success, message)
        # Set reward and done in observation (these are expected by Observation base class)
        observation.reward = reward
        observation.done = self.is_done

        return observation

    @property
    def state(self) -> WarehouseState:
        """Get current episode state."""
        return WarehouseState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            packages_delivered=self.packages_delivered,
            total_packages=self.num_packages,
            difficulty_level=self.difficulty_level,
            grid_size=(self.grid_width, self.grid_height),
            cum_reward=self.cum_reward,
            is_done=self.is_done,
        )

    def _generate_warehouse(self):
        """Generate the warehouse grid layout."""
        # Initialize empty grid
        self.grid = [
            [EMPTY for _ in range(self.grid_width)] for _ in range(self.grid_height)
        ]

        # Add walls around perimeter
        for x in range(self.grid_width):
            self.grid[0][x] = WALL
            self.grid[self.grid_height - 1][x] = WALL
        for y in range(self.grid_height):
            self.grid[y][0] = WALL
            self.grid[y][self.grid_width - 1] = WALL

        # Add random shelves/obstacles
        obstacles_placed = 0
        attempts = 0
        while (
            obstacles_placed < self.num_obstacles and attempts < self.num_obstacles * 10
        ):
            x = random.randint(2, self.grid_width - 3)
            y = random.randint(2, self.grid_height - 3)

            # Don't place near center (robot start)
            if abs(x - self.grid_width // 2) < 2 and abs(y - self.grid_height // 2) < 2:
                attempts += 1
                continue

            if self.grid[y][x] == EMPTY:
                self.grid[y][x] = SHELF
                obstacles_placed += 1

            attempts += 1

        # Create pickup zones (top-left area)
        self.pickup_zones = []
        for _ in range(min(3, self.num_packages)):
            x = random.randint(1, self.grid_width // 3)
            y = random.randint(1, self.grid_height // 3)
            if self.grid[y][x] == EMPTY:
                self.grid[y][x] = PICKUP_ZONE
                self.pickup_zones.append((x, y))

        # Create dropoff zones (bottom-right area)
        self.dropoff_zones = []
        for _ in range(min(3, self.num_packages)):
            x = random.randint(2 * self.grid_width // 3, self.grid_width - 2)
            y = random.randint(2 * self.grid_height // 3, self.grid_height - 2)
            if self.grid[y][x] == EMPTY:
                self.grid[y][x] = DROPOFF_ZONE
                self.dropoff_zones.append((x, y))

    def _generate_packages(self):
        """Generate packages with random pickup/dropoff locations."""
        self.packages = []
        for i in range(self.num_packages):
            pickup_loc = (
                random.choice(self.pickup_zones) if self.pickup_zones else (1, 1)
            )
            dropoff_loc = (
                random.choice(self.dropoff_zones)
                if self.dropoff_zones
                else (self.grid_width - 2, self.grid_height - 2)
            )

            package = Package(
                id=i,
                status="waiting",
                pickup_location=pickup_loc,
                dropoff_location=dropoff_loc,
                priority=random.randint(1, 3),
                time_waiting=0,
            )
            self.packages.append(package)

    def _move_robot(self, direction: int) -> Tuple[bool, str]:
        """
        Move robot in specified direction.

        Args:
            direction: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        Returns:
            (success, message)
        """
        x, y = self.robot_position

        if direction == 0:  # UP
            new_pos = (x, y - 1)
        elif direction == 1:  # DOWN
            new_pos = (x, y + 1)
        elif direction == 2:  # LEFT
            new_pos = (x - 1, y)
        elif direction == 3:  # RIGHT
            new_pos = (x + 1, y)
        else:
            return False, "Invalid direction"

        new_x, new_y = new_pos

        # Check bounds
        if (
            new_x < 0
            or new_x >= self.grid_width
            or new_y < 0
            or new_y >= self.grid_height
        ):
            return False, "Cannot move outside warehouse bounds"

        # Check collision with walls/shelves
        if self.grid[new_y][new_x] in [WALL, SHELF]:
            return False, "Cannot move into obstacle"

        self.robot_position = new_pos
        return True, f"Moved {WarehouseAction.ACTION_NAMES[direction]}"

    def _pickup_package(self) -> Tuple[bool, str]:
        """Attempt to pick up a package."""
        if self.robot_carrying is not None:
            return False, "Robot already carrying a package"

        # Check if at pickup zone
        x, y = self.robot_position
        if self.grid[y][x] != PICKUP_ZONE:
            return False, "Not at a pickup zone"

        # Find available package at this location
        for package in self.packages:
            if package.status == "waiting" and package.pickup_location == (x, y):
                package.status = "picked"
                self.robot_carrying = package.id
                return True, f"Picked up package #{package.id}"

        return False, "No packages available at this location"

    def _dropoff_package(self) -> Tuple[bool, str]:
        """Attempt to drop off a package."""
        if self.robot_carrying is None:
            return False, "Not carrying any package"

        # Check if at dropoff zone
        x, y = self.robot_position
        if self.grid[y][x] != DROPOFF_ZONE:
            return False, "Not at a dropoff zone"

        # Find the package being carried
        package = next((p for p in self.packages if p.id == self.robot_carrying), None)
        if package is None:
            return False, "Package not found"

        # Check if correct dropoff location
        if package.dropoff_location == (x, y):
            package.status = "delivered"
            self.packages_delivered += 1
            self.robot_carrying = None
            return True, f"Successfully delivered package #{package.id}!"
        else:
            return False, f"Wrong dropoff zone for package #{package.id}"

    def _get_observation(
        self, action_success: bool, message: str
    ) -> WarehouseObservation:
        """Create observation object."""
        packages_data = [
            {
                "id": p.id,
                "status": p.status,
                "pickup_location": p.pickup_location,
                "dropoff_location": p.dropoff_location,
                "priority": p.priority,
                "time_waiting": p.time_waiting,
            }
            for p in self.packages
        ]

        return WarehouseObservation(
            grid=self.grid,
            robot_position=self.robot_position,
            robot_carrying=self.robot_carrying,
            packages=packages_data,
            step_count=self.step_count,
            packages_delivered=self.packages_delivered,
            total_packages=self.num_packages,
            time_remaining=self.max_steps - self.step_count,
            action_success=action_success,
            message=message,
        )

    def render_ascii(self) -> str:
        """Render warehouse as ASCII art."""
        symbols = {
            EMPTY: ".",
            WALL: "█",
            SHELF: "#",
            PICKUP_ZONE: "P",
            DROPOFF_ZONE: "D",
        }

        lines = []
        lines.append("=" * (self.grid_width * 2 + 1))
        lines.append(
            f"Step: {self.step_count}/{self.max_steps} | Delivered: {self.packages_delivered}/{self.num_packages} | Reward: {self.cum_reward:.1f}"
        )
        lines.append("=" * (self.grid_width * 2 + 1))

        for y in range(self.grid_height):
            row = ""
            for x in range(self.grid_width):
                if (x, y) == self.robot_position:
                    if self.robot_carrying is not None:
                        row += "R "  # Robot carrying package
                    else:
                        row += "r "  # Robot empty
                else:
                    row += symbols[self.grid[y][x]] + " "
            lines.append(row)

        lines.append("=" * (self.grid_width * 2 + 1))
        lines.append(f"Robot at {self.robot_position}, carrying: {self.robot_carrying}")

        # Show package info
        for package in self.packages:
            status_icon = (
                "✓"
                if package.status == "delivered"
                else ("↻" if package.status == "picked" else "○")
            )
            lines.append(
                f"{status_icon} Package #{package.id}: {package.status} (P{package.pickup_location}→D{package.dropoff_location})"
            )

        lines.append("=" * (self.grid_width * 2 + 1))
        lines.append(
            "Legend: r/R=Robot(empty/carrying), P=Pickup, D=Dropoff, #=Shelf, █=Wall"
        )

        return "\n".join(lines)
