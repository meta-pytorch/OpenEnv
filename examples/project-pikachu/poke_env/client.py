"""
Pokemon Battle Environment HTTP Client.

This module provides the client for connecting to a Pokemon Battle Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.client_types import StepResult
from core.http_env_client import HTTPEnvClient

from .models import PokemonAction, PokemonObservation, PokemonState, PokemonData

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class PokemonEnv(HTTPEnvClient[PokemonAction, PokemonObservation]):
    """
    HTTP client for Pokemon Battle Environment.

    This client connects to a Pokemon Battle Environment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = PokemonEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.active_pokemon.species)
        >>>
        >>> # Take an action
        >>> result = client.step(PokemonAction(action_type="move", action_index=0))
        >>> print(result.reward, result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = PokemonEnv.from_docker_image("pokemon-env:latest")
        >>> result = client.reset()
        >>> result = client.step(PokemonAction(action_type="switch", action_index=1))
    """

    def _step_payload(self, action: PokemonAction) -> Dict[str, Any]:
        """
        Convert PokemonAction to JSON payload for step request.

        Args:
            action: PokemonAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "action_type": action.action_type,
            "action_index": action.action_index,
            "move_id": action.move_id,
            "switch_pokemon": action.switch_pokemon,
            "mega_evolve": action.mega_evolve,
            "dynamax": action.dynamax,
            "terastallize": action.terastallize,
        }

    def _parse_pokemon_data(self, data: Dict[str, Any]) -> PokemonData:
        """Parse Pokemon data from JSON."""
        return PokemonData(
            species=data.get("species", "unknown"),
            hp_percent=data.get("hp_percent", 0.0),
            max_hp=data.get("max_hp", 100),
            current_hp=data.get("current_hp", 0),
            level=data.get("level", 50),
            status=data.get("status"),
            types=data.get("types", []),
            ability=data.get("ability"),
            item=data.get("item"),
            attack=data.get("attack", 0),
            defense=data.get("defense", 0),
            special_attack=data.get("special_attack", 0),
            special_defense=data.get("special_defense", 0),
            speed=data.get("speed", 0),
            boosts=data.get("boosts", {}),
            moves=data.get("moves", []),
            fainted=data.get("fainted", False),
            active=data.get("active", False),
        )

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PokemonObservation]:
        """
        Parse server response into StepResult[PokemonObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with PokemonObservation.
        """
        obs_data = payload.get("observation", {})

        active_pokemon = None
        if obs_data.get("active_pokemon"):
            active_pokemon = self._parse_pokemon_data(obs_data["active_pokemon"])

        opponent_active = None
        if obs_data.get("opponent_active_pokemon"):
            opponent_active = self._parse_pokemon_data(obs_data["opponent_active_pokemon"])

        team = [self._parse_pokemon_data(p) for p in obs_data.get("team", [])]
        opponent_team = [self._parse_pokemon_data(p) for p in obs_data.get("opponent_team", [])]

        observation = PokemonObservation(
            active_pokemon=active_pokemon,
            opponent_active_pokemon=opponent_active,
            team=team,
            opponent_team=opponent_team,
            available_moves=obs_data.get("available_moves", []),
            available_switches=obs_data.get("available_switches", []),
            legal_actions=obs_data.get("legal_actions", []),
            field_conditions=obs_data.get("field_conditions", {}),
            turn=obs_data.get("turn", 0),
            forced_switch=obs_data.get("forced_switch", False),
            can_mega_evolve=obs_data.get("can_mega_evolve", False),
            can_dynamax=obs_data.get("can_dynamax", False),
            can_terastallize=obs_data.get("can_terastallize", False),
            battle_format=obs_data.get("battle_format", "gen8randombattle"),
            battle_id=obs_data.get("battle_id"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PokemonState:
        """
        Parse server response into PokemonState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            PokemonState object with environment state information.
        """
        return PokemonState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            battle_format=payload.get("battle_format", "gen8randombattle"),
            player_username=payload.get("player_username", "player"),
            server_url=payload.get("server_url", "localhost:8000"),
            battle_id=payload.get("battle_id"),
            is_battle_finished=payload.get("is_battle_finished", False),
            battle_winner=payload.get("battle_winner"),
        )
