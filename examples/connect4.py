import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from connect4_env import Connect4Action, Connect4Env


def render_connect4_board(board, ax, player_colors={1: "red", 2: "yellow", -1: "yellow"}, show=True):
    """
    Render a Connect 4 board using matplotlib.

    Args:
        board: 2D list, numpy array, or board object (6x7) with values:
               0 -> empty, 1 -> player 1, 2 -> player 2 (or -1 for player 2)
        player_colors: dict mapping player numbers to colors.
        show: If True, calls plt.show(). If False, returns the figure.

    Returns:
        The matplotlib figure and axis (if show=False).
    """
    # Extract board data if it's an object with board attribute
    if hasattr(board, 'board'):
        b_map = np.array(board.board)
    elif hasattr(board, '__array__'):
        b_map = np.array(board)
    else:
        b_map = np.array(board)
    
    # Handle different player value representations
    # Some environments use 1 and 2, others use 1 and -1
    rows, cols = b_map.shape

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw the blue board background
    rect = plt.Rectangle((0, 0), cols, rows, color="#0055FF", zorder=0)
    ax.add_patch(rect)

    # Draw circular holes
    for r in range(rows):
        for c in range(cols):
            center = (c + 0.5, rows - 1 - r + 0.5)  # Fixed: removed extra -1
            val = b_map[r, c]
            
            # Handle different value representations
            if val == 1:
                color = player_colors[1]
            elif val == 2 or val == -1:
                color = player_colors.get(2, player_colors.get(-1, "yellow"))
            else:
                color = "white"

            circ = Circle(center, 0.4, color=color, ec="black", lw=1.5)
            ax.add_patch(circ)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return ax
    

def main(render=True):
    print("Connecting to Connect4 environment...")
    env = Connect4Env(base_url="http://localhost:8000")

    try:
        print("\nResetting environment...")
        result = env.reset()

        frames = []
        rewards = []
        steps = []

        # Collect all frames
        board = np.array(result.observation.board).reshape(6, 7)
        frames.append(board.copy())
        rewards.append(result.reward or 0)
        steps.append(0)

        for step in range(100):
            if result.done:
                break

            action_id = int(np.random.choice(result.observation.legal_actions))
            result = env.step(Connect4Action(column=action_id))

            board = np.array(result.observation.board).reshape(6, 7)
            frames.append(board.copy())
            rewards.append(result.reward or 0)
            steps.append(step + 1)

            if result.done:
                print(f"Game finished at step {step + 1} with reward {result.reward}")
                break
        
        if render:
            # Create a single figure and update it
            fig, ax = plt.subplots(figsize=(7, 6))
            
            def animate_frame(i):
                ax.clear()
                # Use the render function but don't show immediately
                render_connect4_board(frames[i], ax=ax, show=False)
                ax.set_title(f"Step: {steps[i]}, Reward: {rewards[i]:.2f}\nTotal: {sum(rewards[:i+1]):.2f}", 
                            fontsize=12, pad=20)
                return ax.patches
            
            # Create animation
            ani = FuncAnimation(fig, animate_frame, frames=len(frames), 
                            interval=700, repeat=False, blit=False)
            
            plt.tight_layout()
            plt.show(block=True)

    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main(render=True)
