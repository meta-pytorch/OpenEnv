"""
Pokemon Battle Environment Server Implementation.

This module provides a properly synchronized bridge between poke-env's async
battle system and OpenEnv's HTTP-based Environment interface.

Key Design:
- poke-env runs on dedicated POKE_LOOP background thread
- FastAPI runs on main uvicorn event loop
- Proper synchronization via asyncio.Future and threading primitives
- Handles illegal moves, forced switches, and edge cases
- Supports team preview, mega evolution, dynamax, terastallize
"""

import asyncio
import logging
import uuid
from dataclasses import asdict
from threading import Event, Lock
from typing import Any, Dict, List, Optional

from core.env_server import Action, Environment, Observation

from ..models import PokemonAction, PokemonObservation, PokemonData, PokemonState

try:
    from poke_env.player import Player, RandomPlayer
    from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
    from poke_env import AccountConfiguration, LocalhostServerConfiguration
    from poke_env.concurrency import POKE_LOOP, handle_threaded_coroutines
except ImportError as e:
    raise ImportError(
        "poke-env is not installed. "
        "Please install it with: pip install poke-env"
    ) from e


logger = logging.getLogger(__name__)


class OpenEnvPokemonPlayer(Player):
    """
    Custom Player class for OpenEnv integration.

    This player bridges external action control with poke-env's async battle system.
    Uses proper synchronization between event loops.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Action synchronization (all accessed from POKE_LOOP)
        self._next_action: Optional[PokemonAction] = None
        self._action_event: Optional[asyncio.Event] = None
        self._turn_complete_event: Optional[asyncio.Event] = None

        # Error tracking
        self._last_error: Optional[str] = None
        self._illegal_action_count = 0

    def _ensure_events(self):
        """Ensure events are created on POKE_LOOP."""
        if self._action_event is None:
            self._action_event = asyncio.Event()
        if self._turn_complete_event is None:
            self._turn_complete_event = asyncio.Event()

    def set_next_action(self, action: PokemonAction):
        """
        Set the next action to be executed (called from any thread).

        This schedules the action setting on POKE_LOOP and returns immediately.
        """
        async def _set_action():
            self._ensure_events()  # Ensure events exist on POKE_LOOP
            self._next_action = action
            self._last_error = None
            self._action_event.set()

        # Schedule on POKE_LOOP from any thread
        asyncio.run_coroutine_threadsafe(_set_action(), POKE_LOOP)

    async def wait_for_turn_complete(self, timeout: float = 30.0):
        """Wait for the current turn to complete."""
        self._ensure_events()  # Ensure events exist on POKE_LOOP
        self._turn_complete_event.clear()
        try:
            await asyncio.wait_for(self._turn_complete_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Turn completion timed out after {timeout}s")
            raise

    async def choose_move(self, battle):
        """
        Choose a move based on the externally provided action.

        Waits for an action to be set via set_next_action(), validates it,
        and executes it. Handles illegal moves by retrying with random move.
        """
        self._ensure_events()  # Ensure events exist on POKE_LOOP

        # Wait for action with timeout
        try:
            await asyncio.wait_for(self._action_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.error("Action timeout - no action received in 60s")
            self._last_error = "Action timeout"
            return ForfeitBattleOrder()

        action = self._next_action
        self._next_action = None
        self._action_event.clear()

        if action is None:
            logger.warning("No action available, choosing random")
            return self.choose_random_move(battle)

        # Signal turn complete when this method returns
        def signal_complete():
            self._turn_complete_event.set()

        # Parse and execute action
        try:
            order = self._action_to_order(action, battle)
            # Schedule signal for after this coroutine completes
            asyncio.get_event_loop().call_soon(signal_complete)
            return order
        except Exception as e:
            logger.error(f"Error converting action to order: {e}")
            self._last_error = str(e)
            self._illegal_action_count += 1
            asyncio.get_event_loop().call_soon(signal_complete)
            return self.choose_random_move(battle)

    def _action_to_order(self, action: PokemonAction, battle) -> BattleOrder:
        """Convert PokemonAction to BattleOrder, with validation."""

        # Handle forfeit
        if action.action_type == "forfeit":
            return ForfeitBattleOrder()

        # Handle move action
        if action.action_type == "move":
            if not battle.available_moves:
                raise ValueError("No moves available")

            if action.action_index >= len(battle.available_moves):
                raise ValueError(
                    f"Move index {action.action_index} out of range "
                    f"(only {len(battle.available_moves)} moves available)"
                )

            move = battle.available_moves[action.action_index]

            # Check for special mechanics
            if action.mega_evolve and not battle.can_mega_evolve:
                logger.warning("Cannot mega evolve - ignoring flag")
                action.mega_evolve = False

            if action.dynamax and not battle.can_dynamax:
                logger.warning("Cannot dynamax - ignoring flag")
                action.dynamax = False

            if action.terastallize and not battle.can_tera:
                logger.warning("Cannot terastallize - ignoring flag")
                action.terastallize = False

            return self.create_order(
                move,
                mega=action.mega_evolve,
                dynamax=action.dynamax,
                terastallize=action.terastallize,
            )

        # Handle switch action
        elif action.action_type == "switch":
            if not battle.available_switches:
                raise ValueError("No switches available")

            if action.action_index >= len(battle.available_switches):
                raise ValueError(
                    f"Switch index {action.action_index} out of range "
                    f"(only {len(battle.available_switches)} switches available)"
                )

            pokemon = battle.available_switches[action.action_index]
            return self.create_order(pokemon)

        # Handle default action
        elif action.action_type == "default":
            return self.choose_random_move(battle)

        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

    async def teampreview(self, battle):
        """
        Handle team preview phase.

        For now, uses default ordering. Can be extended to accept
        team preview action from client.
        """
        # Default ordering (1-6)
        return "/team 123456"


class PokemonEnvironment(Environment):
    """
    Pokemon Battle Environment for OpenEnv.

    Properly bridges poke-env's async battle system with OpenEnv's sync
    HTTP interface. Handles:
    - Event loop synchronization
    - Action queuing and turn completion
    - Battle state serialization
    - Error handling and illegal moves
    - Reward computation (sparse or dense)

    Args:
        battle_format: Battle format (e.g., "gen9randombattle", "gen9ou")
        player_username: Username for player
        opponent: Opponent player (defaults to RandomPlayer)
        reward_mode: "sparse" (only at end) or "dense" (per-turn shaping)
        max_turns: Maximum turns before auto-forfeit

    Example:
        >>> env = PokemonEnvironment(battle_format="gen9randombattle")
        >>> obs = env.reset()
        >>> print(obs.active_pokemon.species)
        >>> obs = env.step(PokemonAction(action_type="move", action_index=0))
    """

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        player_username: Optional[str] = None,
        opponent: Optional[Player] = None,
        reward_mode: str = "sparse",
        max_turns: int = 1000,
    ):
        """Initialize Pokemon battle environment."""
        super().__init__()

        self.battle_format = battle_format
        self.player_username = player_username or f"player_{uuid.uuid4().hex[:8]}"
        self.reward_mode = reward_mode
        self.max_turns = max_turns

        # Initialize player on POKE_LOOP
        logger.info(f"Creating player {self.player_username} for format {battle_format}")

        self.player = OpenEnvPokemonPlayer(
            account_configuration=AccountConfiguration(self.player_username, None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=battle_format,
            max_concurrent_battles=1,  # One battle at a time
        )

        # Create opponent
        if opponent is None:
            opponent_username = f"opponent_{uuid.uuid4().hex[:8]}"
            logger.info(f"Creating random opponent {opponent_username}")
            self.opponent = RandomPlayer(
                account_configuration=AccountConfiguration(opponent_username, None),
                server_configuration=LocalhostServerConfiguration,
                battle_format=battle_format,
                max_concurrent_battles=1,
            )
        else:
            self.opponent = opponent

        # State
        self._state = PokemonState(
            battle_format=battle_format,
            player_username=self.player_username,
            server_url="localhost:8000",
        )

        # Battle tracking
        self._current_battle = None
        self._battle_future: Optional[asyncio.Future] = None

        # Synchronization
        self._reset_lock = Lock()
        self._step_lock = Lock()

        # Reward tracking (for dense rewards)
        self._last_opponent_fainted = 0
        self._last_player_fainted = 0
        self._last_opponent_hp = 1.0

    def _pokemon_to_data(self, pokemon) -> Optional[PokemonData]:
        """Convert poke-env Pokemon to PokemonData."""
        if pokemon is None:
            return None

        # Extract moves
        moves = []
        for move_id, move in pokemon.moves.items():
            moves.append({
                "id": move_id,
                "type": str(move.type) if hasattr(move, 'type') and move.type else "unknown",
                "power": move.base_power if hasattr(move, 'base_power') else 0,
                "pp": move.current_pp if hasattr(move, 'current_pp') else 0,
                "accuracy": move.accuracy if hasattr(move, 'accuracy') else 1.0,
                "category": str(move.category) if hasattr(move, 'category') else "status",
            })

        # Get base stats
        base_stats = pokemon.base_stats if hasattr(pokemon, 'base_stats') else {}

        # Get current HP
        hp_fraction = pokemon.current_hp_fraction if hasattr(pokemon, 'current_hp_fraction') else 1.0
        max_hp = pokemon.max_hp if (hasattr(pokemon, 'max_hp') and pokemon.max_hp) else 100
        current_hp = int(hp_fraction * max_hp)

        return PokemonData(
            species=pokemon.species if hasattr(pokemon, 'species') else "unknown",
            hp_percent=hp_fraction,
            max_hp=max_hp,
            current_hp=current_hp,
            level=pokemon.level if hasattr(pokemon, 'level') else 50,
            status=str(pokemon.status.name) if (hasattr(pokemon, 'status') and pokemon.status) else None,
            types=[str(t.name) if hasattr(t, 'name') else str(t) for t in (pokemon.types if hasattr(pokemon, 'types') and pokemon.types else [])],
            ability=pokemon.ability if hasattr(pokemon, 'ability') else None,
            item=pokemon.item if hasattr(pokemon, 'item') else None,
            attack=base_stats.get("atk", 0) if isinstance(base_stats, dict) else 0,
            defense=base_stats.get("def", 0) if isinstance(base_stats, dict) else 0,
            special_attack=base_stats.get("spa", 0) if isinstance(base_stats, dict) else 0,
            special_defense=base_stats.get("spd", 0) if isinstance(base_stats, dict) else 0,
            speed=base_stats.get("spe", 0) if isinstance(base_stats, dict) else 0,
            boosts=dict(pokemon.boosts) if hasattr(pokemon, 'boosts') and pokemon.boosts else {},
            moves=moves,
            fainted=pokemon.fainted if hasattr(pokemon, 'fainted') else False,
            active=pokemon.active if hasattr(pokemon, 'active') else False,
        )

    def _extract_field_conditions(self, battle) -> Dict[str, Any]:
        """Extract field conditions from battle state."""
        conditions = {}

        # Weather
        if hasattr(battle, 'weather') and battle.weather:
            for weather, turn_started in battle.weather.items():
                conditions["weather"] = str(weather.name)
                conditions["weather_turn"] = turn_started
                break  # Only one weather active

        # Terrain/Fields
        if hasattr(battle, 'fields') and battle.fields:
            terrains = []
            for field, turn_started in battle.fields.items():
                terrains.append({
                    "name": str(field.name),
                    "turn_started": turn_started
                })
            conditions["terrains"] = terrains

        # Side conditions (your side)
        if hasattr(battle, 'side_conditions'):
            side_conds = {}
            for condition, value in battle.side_conditions.items():
                side_conds[str(condition.name)] = value
            conditions["side_conditions"] = side_conds

        # Opponent side conditions
        if hasattr(battle, 'opponent_side_conditions'):
            opp_side_conds = {}
            for condition, value in battle.opponent_side_conditions.items():
                opp_side_conds[str(condition.name)] = value
            conditions["opponent_side_conditions"] = opp_side_conds

        return conditions

    def _compute_reward(self, battle, done: bool) -> float:
        """Compute reward based on reward_mode."""

        if self.reward_mode == "sparse":
            # Only reward at end
            if not done:
                return 0.0

            if battle.won:
                return 1.0
            elif battle.lost:
                return -1.0
            else:
                return 0.0  # Tie

        elif self.reward_mode == "dense":
            # Per-turn reward shaping
            reward = 0.0

            # Reward for fainting opponent Pokemon
            opponent_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
            new_faint_count = opponent_fainted - self._last_opponent_fainted
            reward += new_faint_count * 0.2
            self._last_opponent_fainted = opponent_fainted

            # Penalty for losing own Pokemon
            player_fainted = sum(1 for p in battle.team.values() if p.fainted)
            new_player_faint = player_fainted - self._last_player_fainted
            reward -= new_player_faint * 0.2
            self._last_player_fainted = player_fainted

            # Small reward for opponent HP damage
            if battle.opponent_active_pokemon and hasattr(battle.opponent_active_pokemon, 'current_hp_fraction'):
                current_hp = battle.opponent_active_pokemon.current_hp_fraction
                if current_hp is not None:
                    hp_delta = self._last_opponent_hp - current_hp
                    reward += hp_delta * 0.05
                    self._last_opponent_hp = current_hp

            # Final outcome bonus
            if done:
                if battle.won:
                    reward += 0.5
                elif battle.lost:
                    reward -= 0.5

            return reward

        else:
            # Unknown mode, use sparse
            return self._compute_reward(battle, done) if done else 0.0

    def _battle_to_observation(
        self,
        battle,
        reward: Optional[float] = None,
        done: bool = False
    ) -> PokemonObservation:
        """Convert poke-env Battle to PokemonObservation."""

        # Convert Pokemon
        active_pokemon = self._pokemon_to_data(battle.active_pokemon)
        opponent_active = self._pokemon_to_data(battle.opponent_active_pokemon)

        team = [self._pokemon_to_data(p) for p in battle.team.values()]
        opponent_team = [self._pokemon_to_data(p) for p in battle.opponent_team.values()]

        # Available actions
        available_moves = list(range(len(battle.available_moves)))
        available_switches = list(range(len(battle.available_switches)))

        # Build legal actions list
        legal_actions = []
        for i in available_moves:
            legal_actions.append({"type": "move", "index": i})
        for i in available_switches:
            legal_actions.append({"type": "switch", "index": i})

        # Field conditions
        field_conditions = self._extract_field_conditions(battle)

        # Compute reward
        if reward is None:
            reward = self._compute_reward(battle, done)

        return PokemonObservation(
            active_pokemon=active_pokemon,
            opponent_active_pokemon=opponent_active,
            team=team,
            opponent_team=opponent_team,
            available_moves=available_moves,
            available_switches=available_switches,
            legal_actions=legal_actions,
            field_conditions=field_conditions,
            turn=battle.turn,
            forced_switch=battle.force_switch if hasattr(battle, 'force_switch') else False,
            can_mega_evolve=battle.can_mega_evolve if hasattr(battle, 'can_mega_evolve') else False,
            can_dynamax=battle.can_dynamax if hasattr(battle, 'can_dynamax') else False,
            can_terastallize=battle.can_tera if hasattr(battle, 'can_tera') else False,
            battle_format=self.battle_format,
            battle_id=battle.battle_tag if hasattr(battle, 'battle_tag') else None,
            done=done,
            reward=reward,
        )

    def reset(self) -> Observation:
        """
        Reset the environment and start a new battle.

        This method:
        1. Starts a new battle on POKE_LOOP
        2. Waits for battle to initialize
        3. Returns initial observation

        Returns:
            Initial observation for the agent.
        """
        with self._reset_lock:
            logger.info("Resetting Pokemon environment")

            # Reset reward tracking
            self._last_opponent_fainted = 0
            self._last_player_fainted = 0
            self._last_opponent_hp = 1.0

            # Start battle on POKE_LOOP
            async def start_battle():
                """Start a single battle and return when it's initialized."""
                logger.info("Starting battle...")

                # Use battle_against which returns when battle is complete
                # We need to start it but not wait for completion
                battle_task = asyncio.create_task(
                    self.player.battle_against(self.opponent, n_battles=1)
                )

                # Wait for battle to be created (not completed)
                max_wait = 10.0  # 10 seconds
                start_time = asyncio.get_event_loop().time()

                while asyncio.get_event_loop().time() - start_time < max_wait:
                    if self.player.battles:
                        # Battle has started!
                        break
                    await asyncio.sleep(0.1)

                if not self.player.battles:
                    raise TimeoutError("Battle did not start within 10 seconds")

                logger.info(f"Battle started: {list(self.player.battles.keys())}")
                return battle_task

            # Run on POKE_LOOP
            future = asyncio.run_coroutine_threadsafe(start_battle(), POKE_LOOP)
            try:
                self._battle_future = future.result(timeout=15.0)
            except Exception as e:
                logger.error(f"Failed to start battle: {e}")
                raise RuntimeError(f"Failed to start battle: {e}")

            # Get battle reference
            if not self.player.battles:
                raise RuntimeError("No battle created")

            battle_tag = list(self.player.battles.keys())[0]
            self._current_battle = self.player.battles[battle_tag]

            logger.info(f"Battle initialized: {battle_tag}")

            # Update state
            self._state.episode_id = str(uuid.uuid4())
            self._state.step_count = 0
            self._state.battle_id = battle_tag
            self._state.is_battle_finished = False
            self._state.battle_winner = None

            # Return initial observation
            return self._battle_to_observation(self._current_battle, reward=None, done=False)

    def step(self, action: Action) -> Observation:
        """
        Execute agent's action and wait for turn completion.

        This method:
        1. Validates action type
        2. Sends action to player
        3. Waits for turn to complete
        4. Returns updated observation

        Args:
            action: PokemonAction specifying move or switch

        Returns:
            Observation after executing the action.
        """
        with self._step_lock:
            if not isinstance(action, PokemonAction):
                raise TypeError(f"Expected PokemonAction, got {type(action)}")

            if self._current_battle is None:
                raise RuntimeError("No active battle. Call reset() first.")

            logger.debug(f"Step: action={action.action_type}, index={action.action_index}")

            # Send action to player (schedules on POKE_LOOP)
            self.player.set_next_action(action)

            # Wait for turn to complete on POKE_LOOP
            async def wait_turn():
                await self.player.wait_for_turn_complete(timeout=30.0)

            future = asyncio.run_coroutine_threadsafe(wait_turn(), POKE_LOOP)
            try:
                future.result(timeout=35.0)
            except Exception as e:
                logger.error(f"Error waiting for turn: {e}")
                # Continue anyway - battle may have ended

            # Update state
            self._state.step_count += 1

            # Check if battle is done
            done = self._current_battle.finished

            if done:
                self._state.is_battle_finished = True
                if self._current_battle.won:
                    self._state.battle_winner = self.player_username
                    logger.info("Battle won!")
                elif self._current_battle.lost:
                    self._state.battle_winner = "opponent"
                    logger.info("Battle lost!")
                else:
                    self._state.battle_winner = "tie"
                    logger.info("Battle tied!")

            # Check for max turns
            if self._state.step_count >= self.max_turns and not done:
                logger.warning(f"Max turns ({self.max_turns}) reached, forcing forfeit")
                done = True

            # Return observation
            obs = self._battle_to_observation(self._current_battle, reward=None, done=done)

            # Add error info if available
            if self.player._last_error:
                obs.metadata["last_error"] = self.player._last_error
                obs.metadata["illegal_action_count"] = self.player._illegal_action_count

            return obs

    def close(self):
        """Clean up resources."""
        logger.info("Closing Pokemon environment")

        # Cancel battle if running
        if self._battle_future and not self._battle_future.done():
            self._battle_future.cancel()

        # Note: We don't close POKE_LOOP as it's global and shared

    @property
    def state(self) -> PokemonState:
        """Get current environment state."""
        return self._state
