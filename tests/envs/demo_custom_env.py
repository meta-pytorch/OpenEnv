"""Demo script to see the custom BrowserGym environment in action.

This runs with a visible browser window so you can see the task and actions.

Run this:
    source .venv/bin/activate
    python3 tests/envs/demo_custom_env.py
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from envs.browsergym_env.server.browsergym_environment import BrowserGymEnvironment
from browsergym_env.models import BrowserGymAction

print("=" * 70)
print("Custom BrowserGym Environment - Visual Demo")
print("=" * 70)
print("\nOpening browser window with custom task...\n")

# Create environment with visible browser
env = BrowserGymEnvironment(
    benchmark="custom",
    task_name="copy-paste",
    headless=False,
    viewport_width=1280,
    viewport_height=720,
)

# Reset and show initial state
print("Browser opened, loading task...")
obs = env.reset()
print(f"\nGoal: {obs.goal}")
print("Initial page loaded.\n")
time.sleep(3)  # Give you time to see the page

# Execute some actions step-by-step
print("Now executing actions (watch the browser):\n")

actions = [
    ("Click on source text field", "click('#source-text')"),
    ("Select all text (Ctrl+A)", "press('Control+A')"),
    ("Copy text (Ctrl+C)", "press('Control+C')"),
    ("Click on target field", "click('#target-text')"),
    ("Paste text (Ctrl+V)", "press('Control+V')"),
    ("Click submit button", "click('#submit-btn')"),
]

for i, (description, action_str) in enumerate(actions, 1):
    print(f"Step {i}: {description}")
    action = BrowserGymAction(action_str=action_str)
    obs = env.step(action)
    print(f"  Reward: {obs.reward}, Done: {obs.done}")
    
    time.sleep(2)
    
    if obs.done:
        print("\nTask completed.")
        print(f"Total reward: {env.state.cum_reward}")
        break

print("\nKeeping browser open for 5 seconds...")
time.sleep(5)

# Cleanup
env.close()
print("\nBrowser closed. Demo complete.")
