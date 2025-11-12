import gymnasium as gym
from gymnasium.envs.registration import register

class CopyPasteTask(gym.Env):
    """Local mock BrowserGym copy-paste task (for testing)."""
    metadata = {"render_modes": []}

    def __init__(self, **kwargs):  # Accept all extra args like headless, viewport, timeout
        super().__init__()
        self.goal = "Copy the text from the first box and paste it into the second."
        self.done = False
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({
            "goal": gym.spaces.Text(256),
            "text": gym.spaces.Text(256)
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        obs = {"goal": self.goal, "text": ""}
        info = {"goal": self.goal}
        return obs, info

    def step(self, action):
        self.done = True
        reward = 1.0 if action == 1 else 0.0
        obs = {"goal": self.goal, "text": "Task completed"}
        info = {"success": True}
        return obs, reward, self.done, False, info


# ✅ Register this environment so Gym can find it
register(
    id="browsergym/custom.copy-paste",
    entry_point="src.envs.browsergym_env.server.custom.register:CopyPasteTask",
)
