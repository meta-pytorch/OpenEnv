"""
Pokemon Battle Environment Server Implementation.

This module wraps poke-env's Player and Battle classes and exposes them
via the OpenEnv Environment interface.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from core.env_server import Action, Environment, Observation

from ..models import PokemonAction, PokemonObservation, PokemonData, PokemonState

try:
    from poke_env.player import Player, RandomPlayer
    from poke_env.battle import Battle, Move
    from poke_env.data import GenData
    from poke_env import AccountConfiguration, ServerConfiguration, LocalhostServerConfiguration
except ImportError as e:
    raise ImportError(
        "poke-env is not installed. "
        "Please install it with: pip install poke-env"
    ) from e


class OpenEnvPokemonPlayer(Player):
    """
    Custom Player class for OpenEnv integration.
    
    This player allows external control of battle decisions through
    the choose_move method, enabling LLM-based strategy execution.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_action: Optional[PokemonAction] = None
        self._action_ready = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    def set_next_action(self, action: PokemonAction):
        """Set the next action to be executed in the battle."""
        self._next_action = action
        self._action_ready.set()
    
    async def choose_move(self, battle: Battle):
        """
        Choose a move based on the externally provided action.
        
        This method waits for an action to be set via set_next_action(),
        then executes it in the battle.
        """
        await asyncio.wait_for(self._action_ready.wait(), timeout=60.0)
        
        action = self._next_action
        self._next_action = None
        self._action_ready.clear()
        
        if action is None:
            return self.choose_random_move(battle)
        
        if action.action_type == "move":
            if action.action_index < len(battle.available_moves):
                move = battle.available_moves[action.action_index]
                if action.mega_evolve and battle.can_mega_evolve:
                    return self.create_order(move, mega=True)
                elif action.dynamax and battle.can_dynamax:
                    return self.create_order(move, dynamax=True)
                elif action.terastallize and battle.can_tera:
                    return self.create_order(move, terastallize=True)
                else:
                    return self.create_order(move)
            else:
                return self.choose_random_move(battle)
        
        elif action.action_type == "switch":
            if action.action_index < len(battle.available_switches):
                switch_target = battle.available_switches[action.action_index]
                return self.create_order(switch_target)
            else:
                return self.choose_random_move(battle)
        
        return self.choose_random_move(battle)


class PokemonEnvironment(Environment):
    """
    Pokemon Battle Environment wrapper for OpenEnv.

    This environment wraps poke-env's battle system and provides a clean
    interface for RL training with Pokemon battles.

    Args:
        battle_format: Battle format to use (e.g., "gen8randombattle", "gen8ou")
        player_username: Username for the player
        server_config: ServerConfiguration for Pokemon Showdown connection
        opponent: Opponent player (defaults to RandomPlayer)

    Example:
        >>> env = PokemonEnvironment(battle_format="gen8randombattle")
        >>> obs = env.reset()
        >>> print(obs.active_pokemon.species)
        >>> obs = env.step(PokemonAction(action_type="move", action_index=0))
        >>> print(obs.reward, obs.done)
    """

    def __init__(
        self,
        battle_format: str = "gen8randombattle",
        player_username: Optional[str] = None,
        server_config: Optional[ServerConfiguration] = None,
        opponent: Optional[Player] = None,
    ):
        """Initialize Pokemon battle environment."""
        super().__init__()

        self.battle_format = battle_format
        self.player_username = player_username or f"player_{uuid.uuid4().hex[:8]}"
        
        if server_config is None:
            server_config = LocalhostServerConfiguration
        
        self.server_config = server_config
        
        self.player = OpenEnvPokemonPlayer(
            account_configuration=AccountConfiguration(self.player_username, None),
            server_configuration=server_config,
            battle_format=battle_format,
        )
        
        if opponent is None:
            opponent_username = f"opponent_{uuid.uuid4().hex[:8]}"
            self.opponent = RandomPlayer(
                account_configuration=AccountConfiguration(opponent_username, None),
                server_configuration=server_config,
                battle_format=battle_format,
            )
        else:
            self.opponent = opponent
        
        self._state = PokemonState(
            battle_format=battle_format,
            player_username=self.player_username,
            server_url=getattr(server_config, 'websocket_url', 'localhost:8000'),
        )
        
        self._current_battle: Optional[Battle] = None
        self._battle_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
    def _pokemon_to_data(self, pokemon) -> Optional[PokemonData]:
        """Convert poke-env Pokemon to PokemonData."""
        if pokemon is None:
            return None
        
        moves = []
        for move_id, move in pokemon.moves.items():
            moves.append({
                "id": move_id,
                "type": str(move.type) if hasattr(move, 'type') and move.type else "unknown",
                "power": move.base_power if hasattr(move, 'base_power') else 0,
                "pp": move.current_pp if hasattr(move, 'current_pp') else 0,
                "accuracy": move.accuracy if hasattr(move, 'accuracy') else 100,
            })
        
        base_stats = pokemon.base_stats if hasattr(pokemon, 'base_stats') else {}
        
        return PokemonData(
            species=pokemon.species if hasattr(pokemon, 'species') else "unknown",
            hp_percent=pokemon.current_hp_fraction if hasattr(pokemon, 'current_hp_fraction') else 1.0,
            max_hp=pokemon.max_hp if hasattr(pokemon, 'max_hp') and pokemon.max_hp else 100,
            current_hp=int((pokemon.current_hp_fraction if hasattr(pokemon, 'current_hp_fraction') else 1.0) * (pokemon.max_hp if hasattr(pokemon, 'max_hp') and pokemon.max_hp else 100)),
            level=pokemon.level if hasattr(pokemon, 'level') else 50,
            status=str(pokemon.status) if hasattr(pokemon, 'status') and pokemon.status else None,
            types=[str(t) for t in (pokemon.types if hasattr(pokemon, 'types') else [])],
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
    
    def _extract_field_conditions(self, battle: Battle) -> Dict[str, Any]:
        """Extract field conditions from battle state."""
        conditions = {
            "weather": str(battle.weather) if hasattr(battle, 'weather') and battle.weather else None,
            "terrain": str(battle.fields) if hasattr(battle, 'fields') and battle.fields else None,
            "trick_room": False,
        }
        
        conditions["side_conditions"] = {}
        if hasattr(battle, 'side_conditions'):
            for condition, value in battle.side_conditions.items():
                conditions["side_conditions"][str(condition)] = value
        
        conditions["opponent_side_conditions"] = {}
        if hasattr(battle, 'opponent_side_conditions'):
            for condition, value in battle.opponent_side_conditions.items():
                conditions["opponent_side_conditions"][str(condition)] = value
        
        return conditions
    
    def _battle_to_observation(self, battle: Battle, reward: Optional[float] = None, done: bool = False) -> PokemonObservation:
        """Convert poke-env Battle to PokemonObservation."""
        
        active_pokemon = self._pokemon_to_data(battle.active_pokemon)
        opponent_active = self._pokemon_to_data(battle.opponent_active_pokemon)
        
        team = [self._pokemon_to_data(p) for p in battle.team.values()]
        opponent_team = [self._pokemon_to_data(p) for p in battle.opponent_team.values()]
        
        available_moves = list(range(len(battle.available_moves)))
        available_switches = list(range(len(battle.available_switches)))
        
        legal_actions = []
        for i in available_moves:
            legal_actions.append({"type": "move", "index": i})
        for i in available_switches:
            legal_actions.append({"type": "switch", "index": i})
        
        field_conditions = self._extract_field_conditions(battle)
        
        if reward is None and done:
            if battle.won:
                reward = 1.0
            elif battle.lost:
                reward = -1.0
            else:
                reward = 0.0
        
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
            forced_switch=battle.force_switch,
            can_mega_evolve=battle.can_mega_evolve,
            can_dynamax=battle.can_dynamax,
            can_terastallize=battle.can_tera if hasattr(battle, 'can_tera') else False,
            battle_format=self.battle_format,
            battle_id=battle.battle_tag,
            done=done,
            reward=reward,
        )

    def reset(self) -> Observation:
        """Reset the environment and start a new battle.

        Returns:
            Initial observation for the agent.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        
        async def start_battle():
            await self.player.battle_against(self.opponent, n_battles=1)
        
        self._battle_task = self._loop.create_task(start_battle())
        
        try:
            self._loop.run_until_complete(asyncio.sleep(0.5))
        except RuntimeError:
            pass
        
        if self.player.battles:
            battle_tag = list(self.player.battles.keys())[0]
            self._current_battle = self.player.battles[battle_tag]
        else:
            return PokemonObservation(
                done=False,
                reward=None,
            )
        
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._state.battle_id = self._current_battle.battle_tag
        self._state.is_battle_finished = False
        self._state.battle_winner = None
        
        return self._battle_to_observation(self._current_battle, reward=None, done=False)

    def step(self, action: Action) -> Observation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: PokemonAction specifying move or switch

        Returns:
            Observation after executing the action.
        """
        if not isinstance(action, PokemonAction):
            raise TypeError(f"Expected PokemonAction, got {type(action)}")
        
        if self._current_battle is None:
            raise RuntimeError("No active battle. Call reset() first.")
        
        self.player.set_next_action(action)
        
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(asyncio.sleep(0.1))
        
        self._state.step_count += 1
        
        done = self._current_battle.finished
        
        if done:
            self._state.is_battle_finished = True
            if self._current_battle.won:
                self._state.battle_winner = self.player_username
            elif self._current_battle.lost:
                self._state.battle_winner = "opponent"
        
        return self._battle_to_observation(self._current_battle, reward=None, done=done)
    
    def close(self):
        """Clean up resources."""
        if self._loop and not self._loop.is_closed():
            self._loop.close()
        
        if self._battle_task and not self._battle_task.done():
            self._battle_task.cancel()
    
    def state(self) -> PokemonState:
        """Get current environment state."""
        return self._state
