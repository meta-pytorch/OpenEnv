
"""
Atari OpenEnv Example

This script demonstrates how to:
1. Connect to an Atari Environment (Pong)
2. Reset the environment
3. Take random actions
4. Visualize the game state (optional, requires matplotlib)
"""
import sys
import os
import time
import numpy as np

# Add src and envs to path if running from root
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("envs"))

try:
    from atari_env import AtariEnv, AtariAction
except ImportError:
    from envs.atari_env import AtariEnv, AtariAction

async def main():
    # 1. Connect to the environment
    # Ensure the server is running: 
    # uvicorn envs.atari_env.server.app:app --host 0.0.0.0 --port 8001
    print("Connecting to Atari Environment at http://localhost:8001...")
    print("💡 View the game visually at: http://localhost:8001/web")
    
    try:
        env = AtariEnv(base_url="http://localhost:8001")
        
        # Connect to the environment
        await env.connect()
        
        # 2. Reset the environment to start a new episode
        print("Resetting environment...")
        result = await env.reset()
        
        print(f"Game: Pong")
        print(f"Observation Shape: {result.observation.screen_shape}")
        print(f"Legal Actions: {result.observation.legal_actions}")
        
        # 3. Game Loop
        print("\nStarting Game Loop (10 steps)...")
        for step in range(10):
            # Pick a random valid action
            # 0: NOOP, 2: UP, 3: DOWN (for Pong)
            action_id = np.random.choice(result.observation.legal_actions)
            
            # Create typed action
            action = AtariAction(action_id=int(action_id))
            
            # Step the environment
            result = await env.step(action)
            
            print(f"Step {step+1}: Action={action_id}, Reward={result.reward}, Done={result.done}")
            
            if result.done:
                print("Episode finished!")
                await env.reset()
                
        print("\nExample complete!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the Atari server is running on port 8001.")
    finally:
        if 'env' in locals():
            await env.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
