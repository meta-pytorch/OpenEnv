from src.core.http_env_client import HTTPEnvClient, StepResult

from src.envs.browsergym_env.server.browsergym_environment import BrowserGymEnvironment
def test_custom_env():
    print("Testing custom BrowserGym environment...")

    # Initialize your environment
    env = BrowserGymEnvironment(benchmark="custom", task_name="copy-paste")

    # Reset to get first observation
    obs = env.reset()

    # Check if goal exists
    print("Goal:", getattr(obs, "goal", "No goal found"))

    # Basic sanity check
    assert "Copy" in getattr(obs, "goal", ""), "Goal text not loaded properly."

    print("Environment initialized successfully!")

if __name__ == "__main__":
    test_custom_env()