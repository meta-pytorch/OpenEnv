"""
Data models for Pokemon Battle Environment.

This module defines the Action, Observation, and State types for Pokemon battles
via poke-env integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from core.env_server import Action, Observation, State


@dataclass
class PokemonAction(Action):
    """
    Action for Pokemon battles.

    Attributes:
        action_type: Type of action - "move" or "switch"
        action_index: Index of the move (0-3) or switch target (0-5)
        move_id: Optional move identifier (e.g., "thunderbolt")
        switch_pokemon: Optional Pokemon to switch to (by species name or index)
        mega_evolve: Whether to mega evolve this turn (if applicable)
        dynamax: Whether to dynamax this turn (if applicable)
        terastallize: Whether to terastallize this turn (if applicable)
    """
    action_type: Literal["move", "switch"] = "move"
    action_index: int = 0
    move_id: Optional[str] = None
    switch_pokemon: Optional[str] = None
    mega_evolve: bool = False
    dynamax: bool = False
    terastallize: bool = False


@dataclass
class PokemonData:
    """Simplified Pokemon data for observations."""
    species: str
    hp_percent: float
    max_hp: int
    current_hp: int
    level: int
    status: Optional[str]
    types: List[str]
    ability: Optional[str]
    item: Optional[str]
    
    attack: int
    defense: int
    special_attack: int
    special_defense: int
    speed: int
    
    boosts: Dict[str, int] = field(default_factory=dict)
    moves: List[Dict[str, Any]] = field(default_factory=list)
    
    fainted: bool = False
    active: bool = False


@dataclass
class PokemonObservation(Observation):
    """
    Observation from Pokemon battle environment.

    This represents the full battle state visible to the agent.

    Attributes:
        active_pokemon: Currently active Pokemon on your side
        opponent_active_pokemon: Currently active opponent Pokemon
        team: Your full team of 6 Pokemon
        opponent_team: Opponent's team (may have limited visibility)
        available_moves: List of move indices you can use (0-3)
        available_switches: List of Pokemon indices you can switch to (0-5)
        legal_actions: Combined list of legal action descriptors
        field_conditions: Dict of field effects (weather, terrain, hazards, etc.)
        turn: Current turn number
        forced_switch: Whether you must switch (active Pokemon fainted)
        can_mega_evolve: Whether mega evolution is possible this turn
        can_dynamax: Whether dynamax is possible this turn
        can_terastallize: Whether terastallization is possible this turn
        battle_format: Battle format (e.g., "gen8randombattle", "gen8ou")
    """
    active_pokemon: Optional[PokemonData] = None
    opponent_active_pokemon: Optional[PokemonData] = None
    team: List[PokemonData] = field(default_factory=list)
    opponent_team: List[PokemonData] = field(default_factory=list)
    
    available_moves: List[int] = field(default_factory=list)
    available_switches: List[int] = field(default_factory=list)
    legal_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    field_conditions: Dict[str, Any] = field(default_factory=dict)
    turn: int = 0
    forced_switch: bool = False
    
    can_mega_evolve: bool = False
    can_dynamax: bool = False
    can_terastallize: bool = False
    
    battle_format: str = "gen8randombattle"
    battle_id: Optional[str] = None


@dataclass
class PokemonState(State):
    """
    State for Pokemon battle environment.

    Attributes:
        battle_format: Battle format being used
        player_username: Player's username
        server_url: Pokemon Showdown server URL
        battle_id: Current battle ID
        is_battle_finished: Whether the battle has concluded
        battle_winner: Winner of the battle (if finished)
    """
    battle_format: str = "gen8randombattle"
    player_username: str = "player"
    server_url: str = "localhost:8000"
    battle_id: Optional[str] = None
    is_battle_finished: bool = False
    battle_winner: Optional[str] = None
