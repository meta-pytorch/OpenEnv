import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from envs.connect4_env import Connect4Action, Connect4Env


def main():
    print("Connecting to Connect4 environment...")
    env = Connect4Env(base_url="http://localhost:8000")

    try:
        print("\nResetting environment...")
        result = env.reset()

        frames = []
        rewards = []
        steps = []

        # store the initial board
        board = np.array(result.observation.board).reshape(6, 7)
        frames.append(board.copy())
        rewards.append(result.reward or 0)
        steps.append(0)

        for step in range(100):
            if result.done:
                break

            # take random legal action
            action_id = int(np.random.choice(result.observation.legal_actions))
            result = env.step(Connect4Action(action_id))

            board = np.array(result.observation.board).reshape(6, 7)
            frames.append(board.copy())
            rewards.append(result.reward or 0)
            steps.append(step + 1)

            if result.done:
                print(f"Game finished at step {step + 1} with reward {result.reward}. Accumulated rewards: {sum(rewards)}")
                break

        # plot animation
        fig, ax = plt.subplots()
        im = ax.imshow(frames[0], cmap='jet', vmin=-1, vmax=1)
        text = ax.text(0.02, 1.02, '', transform=ax.transAxes, color='white', fontsize=12,
                       bbox=dict(facecolor='black', alpha=1, boxstyle='round'))

        def update(i):
            im.set_data(frames[i])
            text.set_text(f"Step: {steps[i]}, Reward: {rewards[i]:.2f} Accumulated rewards: {sum(rewards)}")
            return im, text

        ani = FuncAnimation(fig, update, frames=len(frames), interval=700, repeat=False)
        plt.show(block=True)

    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
