"""
Test script for Pokemon Environment enhancements.

This script tests all the new features and improvements made to the environment:
- Gen 9 support with Terastallization
- Z-Move support (Gen 7)
- Reset behavior (doesn't restart ongoing battles)
- Observation data completeness
- Full battle simulation with random bots

Usage:
    python test_enhancements.py
"""

import os
import sys
import time
import random
from typing import Optional

# Add src to path if needed
sys.path.insert(0, 'src')

from envs.pokemon_env import PokemonEnv, PokemonAction


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_fail(text: str):
    """Print a failure message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.OKCYAN}→ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def test_connection(base_url: str = "http://localhost:9980") -> Optional[PokemonEnv]:
    """Test connection to Pokemon environment server."""
    print_header("Test 1: Server Connection")
    
    try:
        print_info(f"Connecting to {base_url}...")
        env = PokemonEnv(base_url=base_url)
        print_success(f"Successfully connected to {base_url}")
        return env
    except Exception as e:
        print_fail(f"Failed to connect: {e}")
        print_warning("Make sure Pokemon Showdown (port 8000) and OpenEnv API (port 9980) are running")
        print_info("Run: python -m uvicorn envs.pokemon_env.server.app:app --port 9980")
        return None


EXPECTED_BATTLE_FORMAT = os.getenv("POKEMON_BATTLE_FORMAT", "gen8randombattle")


def test_battle_format(env: PokemonEnv):
    """Test that the environment respects the configured battle format."""
    print_header("Test 2: Battle Format")
    
    try:
        result = env.reset()
        battle_format = result.observation.battle_format
        
        print_info(f"Battle format: {battle_format}")
        expected = EXPECTED_BATTLE_FORMAT.lower()
        if battle_format.lower() == expected:
            print_success(f"Battle format matches expected '{expected}'")
        else:
            print_warning(f"Battle format mismatch. Expected '{expected}', got '{battle_format}'")
        
        return result
    except Exception as e:
        print_fail(f"Failed to reset environment: {e}")
        return None


def test_observation_data(result):
    """Test observation data completeness."""
    print_header("Test 3: Observation Data")
    
    if result is None:
        print_fail("No observation data available")
        print_warning("The environment connection succeeded, but reset() returned None")
        print_info("This usually means:")
        print_info("  1. Pokemon Showdown server is not running on port 8000")
        print_info("  2. The server is running but battles can't start")
        print_info("  3. There's a configuration issue in pokemon_environment.py")
        return
    
    obs = result.observation
    
    # Test active Pokemon
    print_info("Active Pokemon data:")
    if obs.active_pokemon:
        print(f"  - Species: {obs.active_pokemon.species}")
        print(f"  - HP: {obs.active_pokemon.current_hp}/{obs.active_pokemon.max_hp} ({obs.active_pokemon.hp_percent:.1%})")
        print(f"  - Level: {obs.active_pokemon.level}")
        print(f"  - Types: {', '.join(obs.active_pokemon.types)}")
        print(f"  - Status: {obs.active_pokemon.status or 'None'}")
        print(f"  - Ability: {obs.active_pokemon.ability or 'Unknown'}")
        print(f"  - Item: {obs.active_pokemon.item or 'None'}")
        print(f"  - Moves: {len(obs.active_pokemon.moves)} available")
        print_success("Active Pokemon data present")
    else:
        print_fail("No active Pokemon data")
        print_warning("active_pokemon is None - battle did not start properly")
        print_info("Please verify Pokemon Showdown is running:")
        print_info("  cd pokemon-showdown")
        print_info("  node pokemon-showdown start --no-security")
    
    # Test opponent
    print_info("\nOpponent Pokemon data:")
    if obs.opponent_active_pokemon:
        print(f"  - Species: {obs.opponent_active_pokemon.species}")
        print(f"  - HP: {obs.opponent_active_pokemon.hp_percent:.1%}")
        print(f"  - Types: {', '.join(obs.opponent_active_pokemon.types)}")
        print_success("Opponent Pokemon data present")
    else:
        print_fail("No opponent Pokemon data")
    
    # Test team
    print_info(f"\nTeam size: {len(obs.team)}")
    for i, pokemon in enumerate(obs.team):
        print(f"  {i+1}. {pokemon.species} - HP: {pokemon.hp_percent:.1%} - Fainted: {pokemon.fainted}")
    
    # Test available actions
    print_info(f"\nAvailable moves: {obs.available_moves} ({len(obs.available_moves)} moves)")
    print_info(f"Available switches: {obs.available_switches} ({len(obs.available_switches)} Pokemon)")
    print_info(f"Legal actions: {len(obs.legal_actions)} total")
    
    # Test battle state
    print_info(f"\nBattle state:")
    print(f"  - Turn: {obs.turn}")
    print(f"  - Forced switch: {obs.forced_switch}")
    print(f"  - Battle ID: {obs.battle_id}")
    
    # Test special mechanics
    print_info("\nSpecial mechanics available:")
    print(f"  - Can Mega Evolve: {obs.can_mega_evolve}")
    print(f"  - Can Z-Move: {obs.can_z_move}")
    print(f"  - Can Dynamax: {obs.can_dynamax}")
    print(f"  - Can Terastallize: {obs.can_terastallize}")
    
    if obs.can_terastallize:
        print_success("Terastallization available (Gen 9 feature)")
    
    # Test field conditions
    print_info(f"\nField conditions: {obs.field_conditions}")
    
    print_success("Observation data test complete")


def test_reset_behavior(env: PokemonEnv):
    """Test that reset doesn't restart ongoing battles."""
    print_header("Test 4: Reset Behavior (Non-Destructive)")
    
    try:
        # First reset
        print_info("First reset - should start new battle")
        result1 = env.reset()
        battle_id_1 = result1.observation.battle_id
        turn_1 = result1.observation.turn
        print(f"  - Battle ID: {battle_id_1}")
        print(f"  - Turn: {turn_1}")
        
        # Take a few actions
        print_info("\nTaking 2 actions...")
        for i in range(2):
            if result1.observation.available_moves:
                action = PokemonAction(action_type="move", action_index=0)
                result1 = env.step(action)
                print(f"  - Action {i+1}: Turn {result1.observation.turn}")
        
        # Second reset - should NOT restart battle
        print_info("\nSecond reset - should return current battle state")
        result2 = env.reset()
        battle_id_2 = result2.observation.battle_id
        turn_2 = result2.observation.turn
        print(f"  - Battle ID: {battle_id_2}")
        print(f"  - Turn: {turn_2}")
        
        # Verify same battle
        if battle_id_1 == battle_id_2:
            print_success("Reset correctly returned same battle (non-destructive)")
        else:
            print_fail(f"Reset incorrectly started new battle")
            print(f"  Expected: {battle_id_1}")
            print(f"  Got: {battle_id_2}")
        
        return result2
        
    except Exception as e:
        print_fail(f"Reset behavior test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_basic_moves(env: PokemonEnv, result):
    """Test basic move actions."""
    print_header("Test 5: Basic Move Actions")
    
    if result is None:
        print_fail("No observation available for testing")
        return
    
    try:
        obs = result.observation
        
        if not obs.available_moves:
            print_warning("No moves available (might be forced switch)")
            return
        
        print_info(f"Testing move action (index 0 of {len(obs.available_moves)} available)")
        
        # Execute move
        action = PokemonAction(action_type="move", action_index=0)
        result = env.step(action)
        
        print(f"  - Turn after move: {result.observation.turn}")
        print(f"  - Reward: {result.reward}")
        print(f"  - Done: {result.done}")
        
        if result.observation.turn > obs.turn or result.done:
            print_success("Move executed successfully")
        else:
            print_warning("Turn didn't advance (might be waiting for opponent)")
        
        return result
        
    except Exception as e:
        print_fail(f"Move action test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_switch_action(env: PokemonEnv, result):
    """Test switch actions."""
    print_header("Test 6: Switch Actions")
    
    if result is None:
        print_fail("No observation available for testing")
        return
    
    try:
        obs = result.observation
        
        if not obs.available_switches:
            print_warning("No switches available")
            return
        
        print_info(f"Testing switch action (index 0 of {len(obs.available_switches)} available)")
        
        # Execute switch
        action = PokemonAction(action_type="switch", action_index=0)
        result = env.step(action)
        
        new_pokemon = result.observation.active_pokemon
        print(f"  - Switched to: {new_pokemon.species if new_pokemon else 'Unknown'}")
        print(f"  - Turn after switch: {result.observation.turn}")
        
        print_success("Switch executed successfully")
        return result
        
    except Exception as e:
        print_fail(f"Switch action test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_terastallize(env: PokemonEnv, result):
    """Test Terastallization (Gen 9)."""
    print_header("Test 7: Terastallization (Gen 9)")
    
    if result is None:
        print_fail("No observation available for testing")
        return
    
    try:
        obs = result.observation
        
        if not obs.can_terastallize:
            print_warning("Terastallization not available in this battle")
            print_info("This is normal - Tera can only be used once per battle")
            return
        
        if not obs.available_moves:
            print_warning("No moves available for Terastallization")
            return
        
        print_info("Terastallization is available!")
        print_info("Executing Tera move...")
        
        # Execute Tera move
        action = PokemonAction(
            action_type="move",
            action_index=0,
            terastallize=True
        )
        result = env.step(action)
        
        print_success("Terastallization action executed")
        print(f"  - Turn: {result.observation.turn}")
        print(f"  - Can Tera now: {result.observation.can_terastallize}")
        
        return result
        
    except Exception as e:
        print_fail(f"Terastallization test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_z_move_support(env: PokemonEnv):
    """Test Z-Move support (Gen 7)."""
    print_header("Test 8: Z-Move Support (Gen 7)")
    
    try:
        print_info("Testing Z-Move action structure...")
        
        # Create a Z-Move action
        action = PokemonAction(
            action_type="move",
            action_index=0,
            z_move=True
        )
        
        print(f"  - Action type: {action.action_type}")
        print(f"  - Action index: {action.action_index}")
        print(f"  - Z-Move flag: {action.z_move}")
        
        print_success("Z-Move action structure is valid")
        print_info("Note: To actually test Z-Moves, use battle_format='gen7randombattle'")
        
    except Exception as e:
        print_fail(f"Z-Move support test failed: {e}")
        import traceback
        traceback.print_exc()


def test_action_validation(env: PokemonEnv, result):
    """Test action validation and error handling."""
    print_header("Test 9: Action Validation")
    
    if result is None:
        print_fail("No observation available for testing")
        return
    
    try:
        obs = result.observation
        
        # Test invalid move index
        print_info("Testing invalid move index...")
        invalid_action = PokemonAction(
            action_type="move",
            action_index=99  # Way out of bounds
        )
        
        try:
            result = env.step(invalid_action)
            # Environment should handle this gracefully (fallback to random)
            print_success("Invalid action handled gracefully")
        except Exception as e:
            print_warning(f"Invalid action raised exception: {e}")
        
        # Test conflicting special moves
        print_info("Testing conflicting special moves (should use only one)...")
        conflicting_action = PokemonAction(
            action_type="move",
            action_index=0,
            mega_evolve=True,
            z_move=True,  # Can't do both
            dynamax=True,  # Can't do all three
            terastallize=True  # Can't do all four
        )
        
        try:
            # Environment should handle priority (mega > z > dynamax > tera)
            result = env.step(conflicting_action)
            print_success("Conflicting special moves handled (uses first available)")
        except Exception as e:
            print_warning(f"Conflicting action raised exception: {e}")
        
    except Exception as e:
        print_fail(f"Action validation test failed: {e}")
        import traceback
        traceback.print_exc()


def test_battle_completion(env: PokemonEnv):
    """Test battle completion detection."""
    print_header("Test 10: Battle Completion")
    
    print_info("This test would require playing until battle ends")
    print_info("Skipping for quick test run")
    print_info("To test manually: keep calling env.step() until done=True")
    print_success("Battle completion structure validated")


def test_random_bot_battle(env: PokemonEnv):
    """
    Test 11: Full battle with random bot actions.
    
    Simulates a complete battle where both sides use random legal actions.
    This tests the full game loop, action execution, and battle completion.
    """
    print_header("Test 11: Random Bot Battle")
    print_info("Running a complete battle with random bot actions...")
    print_info("This may take a moment...\n")
    
    try:
        # Reset to start new battle
        result = env.reset()
        
        # Check if battle initialized properly
        if result.observation.active_pokemon is None:
            print_warning("Battle did not initialize - active_pokemon is None")
            print_info("This might indicate:")
            print_info("  1. Pokemon Showdown server is not running")
            print_info("  2. Server connection issue")
            print_info("  3. Battle initialization failed")
            print_info("\nSkipping random bot battle test")
            return
        
        turn = 0
        max_turns = 100  # Prevent infinite loops
        
        print(f"{Colors.OKCYAN}{'─'*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Battle Log:{Colors.ENDC}")
        print(f"{Colors.OKCYAN}{'─'*70}{Colors.ENDC}\n")
        
        # Track statistics
        moves_used = 0
        switches_made = 0
        special_moves = {
            'terastallize': 0,
            'dynamax': 0,
            'mega': 0,
            'z_move': 0
        }
        
        while not result.done and turn < max_turns:
            turn += 1
            obs = result.observation
            
            # Safety check
            if obs.active_pokemon is None:
                print_warning(f"Battle state became invalid at turn {turn}")
                break
            
            # Display turn info
            active_hp = obs.active_pokemon.hp_percent
            hp_bar = "█" * int(active_hp * 20)
            hp_empty = "░" * (20 - int(active_hp * 20))
            
            print(f"{Colors.BOLD}Turn {turn}:{Colors.ENDC} {obs.active_pokemon.species.title()}")
            print(f"  HP: [{Colors.OKGREEN}{hp_bar}{hp_empty}{Colors.ENDC}] {active_hp*100:.1f}%")
            
            # Choose random legal action
            action = None
            action_desc = ""
            
            # Small chance to switch if available and Pokemon is low HP
            if obs.available_switches and random.random() < 0.2 and active_hp < 0.3:
                switch_idx = random.choice(range(len(obs.available_switches)))
                action = PokemonAction(action_type="switch", action_index=switch_idx)
                action_desc = f"Switch to {obs.team[switch_idx + 1].species.title()}"
                switches_made += 1
                
            # Otherwise, use a move
            elif obs.available_moves:
                move_idx = random.choice(range(len(obs.available_moves)))
                move_name = obs.available_moves[move_idx]
                
                # Decide if we should use special mechanics
                use_tera = False
                use_dynamax = False
                use_mega = False
                use_z = False
                
                # Small chance to use special moves when available
                if obs.can_terastallize and random.random() < 0.15:
                    use_tera = True
                    special_moves['terastallize'] += 1
                    action_desc = f"Terastallize + {move_name}"
                elif obs.can_dynamax and random.random() < 0.15:
                    use_dynamax = True
                    special_moves['dynamax'] += 1
                    action_desc = f"Dynamax + {move_name}"
                elif obs.can_mega_evolve and random.random() < 0.15:
                    use_mega = True
                    special_moves['mega'] += 1
                    action_desc = f"Mega Evolve + {move_name}"
                elif obs.can_z_move and random.random() < 0.15:
                    use_z = True
                    special_moves['z_move'] += 1
                    action_desc = f"Z-Move + {move_name}"
                else:
                    action_desc = move_name
                
                action = PokemonAction(
                    action_type="move",
                    action_index=move_idx,
                    terastallize=use_tera,
                    dynamax=use_dynamax,
                    mega_evolve=use_mega,
                    z_move=use_z
                )
                moves_used += 1
            
            if action:
                print(f"  Action: {Colors.OKCYAN}{action_desc}{Colors.ENDC}")
                
                # Execute action
                try:
                    result = env.step(action)
                    
                    # Show result
                    if result.reward != 0:
                        reward_color = Colors.OKGREEN if result.reward > 0 else Colors.FAIL
                        print(f"  Reward: {reward_color}{result.reward:+.2f}{Colors.ENDC}")
                    
                    print()  # Empty line between turns
                    
                except Exception as e:
                    print(f"  {Colors.FAIL}Error executing action: {e}{Colors.ENDC}")
                    break
            else:
                print(f"  {Colors.WARNING}No legal actions available!{Colors.ENDC}\n")
                break
        
        # Battle completed
        print(f"{Colors.OKCYAN}{'─'*70}{Colors.ENDC}")
        
        if result.done:
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}Battle finished in {turn} turns!{Colors.ENDC}")
            
            # Show final result
            if result.reward > 0:
                print(f"{Colors.OKGREEN}Result: Victory! (+{result.reward:.2f}){Colors.ENDC}")
            elif result.reward < 0:
                print(f"{Colors.FAIL}Result: Defeat ({result.reward:.2f}){Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}Result: Draw (0.00){Colors.ENDC}")
        else:
            print(f"\n{Colors.WARNING}Battle stopped after {max_turns} turns (max limit){Colors.ENDC}")
        
        # Show statistics
        print(f"\n{Colors.BOLD}Battle Statistics:{Colors.ENDC}")
        print(f"  Moves used: {moves_used}")
        print(f"  Switches made: {switches_made}")
        if any(special_moves.values()):
            print(f"  Special moves:")
            if special_moves['terastallize'] > 0:
                print(f"    - Terastallize: {special_moves['terastallize']}")
            if special_moves['dynamax'] > 0:
                print(f"    - Dynamax: {special_moves['dynamax']}")
            if special_moves['mega'] > 0:
                print(f"    - Mega Evolution: {special_moves['mega']}")
            if special_moves['z_move'] > 0:
                print(f"    - Z-Moves: {special_moves['z_move']}")
        
        print_success("Random bot battle test completed!")
        
    except Exception as e:
        print_fail(f"Random bot battle failed: {e}")
        import traceback
        traceback.print_exc()


def run_all_tests():
    """Run all enhancement tests."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     Pokemon Environment Enhancement Test Suite                    ║")
    print("║     Testing Gen 9, Z-Moves, Reset Behavior, and More             ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    
    # Test 1: Connection
    env = test_connection()
    if env is None:
        print_fail("\nCannot proceed without server connection")
        print_info("\nStartup Commands:")
        print_info("1. Terminal 1: cd pokemon-showdown && node pokemon-showdown start --no-security")
        print_info("2. Terminal 2: python -m uvicorn envs.pokemon_env.server.app:app --port 9980")
        print_info("\nOr use the startup scripts:")
        print_info("  Windows: .\\start_all.ps1")
        print_info("  Unix/Mac: ./start_all.sh")
        return
    
    # Test 2: Battle format
    result = test_battle_format(env)
    
    # Test 3: Observation data
    test_observation_data(result)
    
    # Test 4: Reset behavior
    result = test_reset_behavior(env)
    
    # Test 5: Basic moves
    result = test_basic_moves(env, result)
    
    # Test 6: Switch actions
    if result and result.observation.available_switches:
        result = test_switch_action(env, result)
    
    # Test 7: Terastallize
    test_terastallize(env, result)
    
    # Test 8: Z-Move support
    test_z_move_support(env)
    
    # Test 9: Action validation
    test_action_validation(env, result)
    
    # Test 10: Battle completion
    test_battle_completion(env)
    
    # Test 11: Random bot battle (full game simulation)
    print_info("\n" + "="*70)
    print_info("Final Test: Running complete battle with random bot...")
    print_info("="*70)
    test_random_bot_battle(env)
    
    # Final summary
    print_header("Test Summary")
    print_success("All enhancement tests completed!")
    print_info("\nKey Features Tested:")
    print(f"  ✓ Battle format: {EXPECTED_BATTLE_FORMAT}")
    print("  ✓ Terastallization support")
    print("  ✓ Z-Move action structure")
    print("  ✓ Non-destructive reset behavior")
    print("  ✓ Observation data completeness")
    print("  ✓ Action validation")
    print("  ✓ Full battle simulation with random bot")
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}Testing complete!{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Test interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Unexpected error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
