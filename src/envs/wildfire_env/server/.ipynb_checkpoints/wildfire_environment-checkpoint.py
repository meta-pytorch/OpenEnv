import os
import random, uuid
from typing import List
from dataclasses import replace

from core.env_server import Environment
from ..models import WildfireAction, WildfireObservation, WildfireState

# Helpers
DIRS_8 = {
    "N":  (0, -1), "NE": (1, -1), "E":  (1, 0), "SE": (1, 1),
    "S":  (0,  1), "SW": (-1, 1), "W":  (-1, 0), "NW": (-1, -1),
    "CALM": (0, 0),
}

def idx(x: int, y: int, w: int) -> int:
    return y * w + x

def in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


class WildfireEnvironment(Environment):
    """
    Weather-aware wildfire simulation.

    Grid encodings:
      0 = ash (burned out)
      1 = fuel / vegetation
      2 = burning
      3 = firebreak
      4 = watered / damp

    Each step:
      - agent acts (water/break/wait)
      - burning spreads to neighbors with wind + humidity effects
      - burning cells burn for multiple ticks, then become ash
    """

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        base_ignite_prob: float = 0.30,
        wind_bias: float = 0.20,      # kept for compatibility (not directly used in B model)
        diag_factor: float = 0.7,     # kept for compatibility (not directly used in B model)
        humidity: float = 0.25,
        init_sources: int = 2,
        seed: int = 3407,
        max_steps: int = 128,
        water_capacity: int = 20,
        break_capacity: int = 50,
    ):
        super().__init__()

        # --- Env-var overrides (optional) ---
        width     = int(os.environ.get("WILDFIRE_WIDTH", width))
        height    = int(os.environ.get("WILDFIRE_HEIGHT", height))
        humidity  = float(os.environ.get("WILDFIRE_HUMIDITY", humidity))
        forced_wind = os.environ.get("WILDFIRE_WIND", None)

        # Store config
        self.w = width
        self.h = height
        self.base_ignite_prob = base_ignite_prob
        self.wind_bias = wind_bias
        self.diag_factor = diag_factor
        self.init_humidity = humidity
        self.init_sources = init_sources
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.init_water = water_capacity
        self.init_breaks = break_capacity
        self.forced_wind = forced_wind

        # burn lifetime in ticks (balanced model)
        self.burn_lifetime = 3

        self._state = WildfireState()

    # --- Core API ---

    def reset(self) -> WildfireObservation:
        # Start with all fuel
        grid = [1] * (self.w * self.h)

        # Wind (forced if provided)
        if self.forced_wind and self.forced_wind in DIRS_8:
            wind_dir = self.forced_wind
        else:
            wind_dir = self.rng.choice(list(DIRS_8.keys()))

        # Humidity small variation around init
        humidity = min(1.0, max(0.0, self.init_humidity + self.rng.uniform(-0.05, 0.05)))

        # Place initial fires
        for _ in range(self.init_sources):
            x = self.rng.randrange(self.w)
            y = self.rng.randrange(self.h)
            grid[idx(x, y, self.w)] = 2

        self._state = WildfireState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_burned=0,
            total_extinguished=0,
            last_action="reset",
            width=self.w,
            height=self.h,
            wind_dir=wind_dir,
            humidity=humidity,
            remaining_water=self.init_water,
            remaining_breaks=self.init_breaks,
            grid=grid,
        )

        # per-cell burn timers (persist across steps)
        self._state.burn_timers = [0] * (self.w * self.h)

        obs = self._make_observation(reward_hint=0.0)
        return obs

    def step(self, action: WildfireAction) -> WildfireObservation:
        st = self._state

        # Apply agent action
        reward = 0.0
        if action.action == "water" and st.remaining_water > 0 and action.x is not None and action.y is not None:
            reward += self._apply_water(action.x, action.y)
        elif action.action == "break" and st.remaining_breaks > 0 and action.x is not None and action.y is not None:
            reward += self._apply_break(action.x, action.y)
        elif action.action == "wait":
            pass
        else:
            # invalid or no resources
            reward -= 0.05

        # Natural fire dynamics
        newly_burned = self._spread_fire()
        st.total_burned += newly_burned

        # small per-step penalty (encourage faster containment)
        reward -= 0.01

        st.step_count += 1
        st.last_action = action.action

        done = self._is_done()

        if done:
            # reward for saved area
            saved = self._saved_cells()
            reward += 0.5 * (saved / (self.w * self.h))
            # reward if fully extinguished
            if self._burning_count() == 0:
                reward += 0.5

        obs = self._make_observation(reward_hint=reward)
        obs.done = done
        obs.reward = reward
        return obs

    @property
    def state(self) -> WildfireState:
        return self._state

    # --- Internal mechanics ---

    def _apply_water(self, x: int, y: int) -> float:
        st = self._state
        if not in_bounds(x, y, self.w, self.h):
            return -0.05
        i = idx(x, y, self.w)
        reward = 0.0

        if st.grid[i] == 2:
            st.grid[i] = 4
            st.burn_timers[i] = 0
            st.total_extinguished += 1
            reward += 0.2
        elif st.grid[i] == 1:
            st.grid[i] = 4  # dampen
            reward += 0.05
        elif st.grid[i] == 4:
            reward -= 0.01
        else:
            reward -= 0.02

        st.remaining_water -= 1
        return reward

    def _apply_break(self, x: int, y: int) -> float:
        st = self._state
        if not in_bounds(x, y, self.w, self.h):
            return -0.05
        i = idx(x, y, self.w)
        reward = 0.0

        if st.grid[i] in (1, 4):
            st.grid[i] = 3
            st.burn_timers[i] = 0
            reward += 0.1
        elif st.grid[i] == 2:
            st.grid[i] = 3
            st.burn_timers[i] = 0
            reward -= 0.02
        elif st.grid[i] == 3:
            reward -= 0.01
        else:
            reward -= 0.02

        st.remaining_breaks -= 1
        return reward

    def _spread_fire(self) -> int:
        """
        Balanced wildfire spread model:
          - burning cells persist for multiple ticks before turning to ash
          - 8-direction spread (diagonals weaker)
          - wind accelerates in wind direction, weakens upwind
          - humidity suppresses ignition probability
          - water (4) reduces ignition chance and reverts to fuel next tick
        """
        st = self._state
        new_grid = st.grid[:]
        newly_burned = 0

        # 8-neighbor model
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (1, -1), (-1, 1), (1, 1)]
        wx, wy = DIRS_8.get(st.wind_dir, (0, 0))

        base = self.base_ignite_prob
        humidity_factor = (1.0 - st.humidity)

        ignite_flags = [False] * (self.w * self.h)

        # First pass: evaluate ignitions, increment burn timers
        for y in range(self.h):
            for x in range(self.w):
                i = idx(x, y, self.w)
                cell = st.grid[i]

                if cell == 2:  # burning
                    st.burn_timers[i] += 1

                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if not in_bounds(nx, ny, self.w, self.h):
                            continue
                        ni = idx(nx, ny, self.w)
                        target = st.grid[ni]

                        if target not in (1, 4):  # only fuel or damp can ignite
                            continue

                        # Wind multiplier
                        if (dx, dy) == (wx, wy):
                            wind_mult = 2.0
                        elif (dx, dy) == (-wx, -wy):
                            wind_mult = 0.5
                        else:
                            wind_mult = 1.0

                        # Diagonals weaker
                        diag_mult = 0.6 if (dx != 0 and dy != 0) else 1.0

                        p = base * humidity_factor * wind_mult * diag_mult

                        # Damp fuel further reduces spread
                        if target == 4:
                            p *= 0.35

                        p = max(0.0, min(1.0, p))
                        if self.rng.random() < p:
                            ignite_flags[ni] = True

        # Second pass: apply transitions
        for i, cell in enumerate(st.grid):
            if cell == 2:
                # burns for burn_lifetime ticks before turning to ash
                if st.burn_timers[i] >= self.burn_lifetime:
                    new_grid[i] = 0  # ash
                    newly_burned += 1
                else:
                    new_grid[i] = 2  # keep burning
            elif ignite_flags[i] and new_grid[i] in (1, 4):
                new_grid[i] = 2
                st.burn_timers[i] = 0
            elif cell == 4:
                # water effect lasts one tick
                new_grid[i] = 1

        st.grid = new_grid
        return newly_burned

    def _burning_count(self) -> int:
        return sum(1 for v in self._state.grid if v == 2)

    def _saved_cells(self) -> int:
        # cells not turned to ash (includes fuel, burning, break, water)
        return sum(1 for v in self._state.grid if v in (1, 2, 3, 4))

    def _is_done(self) -> bool:
        return self._burning_count() == 0 or self._state.step_count >= self.max_steps

    def _make_observation(self, reward_hint: float = 0.0) -> WildfireObservation:
        st = self._state
        burning = self._burning_count()
        burned = sum(1 for v in st.grid if v == 0)
        return WildfireObservation(
            grid=st.grid[:],
            width=self.w,
            height=self.h,
            step=st.step_count,
            wind_dir=st.wind_dir,
            humidity=st.humidity,
            burning_count=burning,
            burned_count=burned,
            reward_hint=reward_hint,
        )
