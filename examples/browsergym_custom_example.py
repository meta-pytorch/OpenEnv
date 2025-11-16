"""Example usage of custom BrowserGym tasks.

This script demonstrates how to create and use custom tasks with the
BrowserGym environment wrapper in OpenEnv.
"""

import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from envs.browsergym_env.server.browsergym_environment import BrowserGymEnvironment
from envs.browsergym_env.models import BrowserGymAction


def multi_tab_copy_paste_example():
    """Run the multi-tab copy-paste example."""
        
    print("Multi-Tab Copy-Paste Task Example")
    print("-" * 80)
    
    # Create environment
    env = BrowserGymEnvironment(
        benchmark="custom",
        task_name="copy-paste-multitab",
        headless=False,
        viewport_width=1280,
        viewport_height=720,
        timeout=10000.0,
    )
    
    # Reset environment
    obs = env.reset()
    print(f"Goal: {obs.goal}\n")
    
    # Solve the multi-tab task- simulates user actions
    steps = [
        ("Select source text", "click('#source-text')"),
        ("Select all text", "press('Control+A')"),
        ("Copy text", "press('Control+C')"),
        ("Navigate to target page", "click('#open-target-btn')"),
        ("Click target input field", "click('#target-text')"),
        ("Paste text", "press('Control+V')"),
        ("Submit form", "click('#submit-btn')"),
    ]
    
    for i, (description, action_str) in enumerate(steps, 1):
        print(f"Step {i}: {description}")
        action = BrowserGymAction(action_str=action_str)
        obs = env.step(action)
        
        # Show which page we're on
        current_page = "unknown"
        if obs.metadata and 'custom_data' in obs.metadata:
            current_page = obs.metadata['custom_data'].get('current_page', 'unknown')
        
        print(f"  Reward: {obs.reward}, Done: {obs.done}, Page: {current_page}")
        
        # Add delay to see the browser actions
        time.sleep(1)
        
        if obs.done:
            print(f"\n✓ Task completed! Total reward: {env.state.cum_reward}")
            break
    
    env.close()
    print("-" * 80)

def single_tab_copy_paste_example():
    """Run the single-tab copy-paste example."""
    
    print("Custom BrowserGym Task Example: Copy-Paste")
    print("-" * 80)
    
    # Create environment
    env = BrowserGymEnvironment(
        benchmark="custom",
        task_name="copy-paste",
        headless=False,
        viewport_width=1280,
        viewport_height=720,
        timeout=10000.0,
    )
    
    # Reset environment
    obs = env.reset()
    print(f"Goal: {obs.goal}\n")
    
    # Solve the task
    steps = [
        ("Click source text field", "click('#source-text')"),
        ("Select all text", "press('Control+A')"),
        ("Copy text", "press('Control+C')"),
        ("Click target field", "click('#target-text')"),
        ("Paste text", "press('Control+V')"),
        ("Click submit button", "click('#submit-btn')"),
    ]
    
    for i, (description, action_str) in enumerate(steps, 1):
        print(f"Step {i}: {description}")
        action = BrowserGymAction(action_str=action_str)
        obs = env.step(action)
        print(f"  Reward: {obs.reward}, Done: {obs.done}")
        
        # Add delay to see the browser actions
        time.sleep(1)
        
        if obs.done:
            print(f"\n✓ Task completed! Total reward: {env.state.cum_reward}")
            break
    
    env.close()
    print("-" * 80)

def main():
    """Run the custom task example."""
    
    # Run single-tab copy-paste example
    print("Single-Tab Copy-Paste")
    single_tab_copy_paste_example()
    
    time.sleep(3)
    
    # Run multi-tab copy-paste example
    print("\nMulti-Tab Copy-Paste")
    multi_tab_copy_paste_example()
    

if __name__ == "__main__":
    # Run main example
    main()

