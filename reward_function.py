# --- 1. Create the Reward Function Factory (The Closure Fix) ---
# You will need to have these imports in your notebook cell
# from envs.dipg_safety_env.models import DIPGAction
# from requests.exceptions import ConnectionError, ReadTimeout

def create_reward_fn(environment):
    """
    This function takes the live 'env' object and returns a reward function
    that has access to it.
    """
    def get_reward_from_environment(completions, prompts, **kwargs):
        scores = []
        # Loop through the batch of completions from the LLM
        for i, response in enumerate(completions):

            # --- START: DEBUGGING CODE ---
            print("="*80)
            print(f"DEBUG: Preparing to send completion #{i} to the environment:")
            # Use repr() to make special characters like newlines ('\\n') visible
            print(repr(response))
            print("="*80)
            # --- END: DEBUGGING CODE ---

            try:
                # This is the line that calls the server.
                # If the server crashes, the error will happen here.
                result = environment.step(DIPGAction(llm_response=response))
                scores.append(result.reward)

            except (ConnectionError, ReadTimeout) as e:
                # This block will now catch the crash!
                print("\\n" + "!"*80)
                print(f"FATAL: Connection lost while processing completion #{i}.")
                print("This means the Gunicorn server has crashed.")
                print(f"The likely culprit is the completion printed above: {repr(response)}")
                print("Check the server's STDERR logs for the Python traceback to find the root cause.")
                print("!"*80 + "\\n")

                # To prevent the entire training run from stopping, we will
                # assign a large penalty and continue.
                scores.append(-50.0)

                # If you WANTED training to stop, you would uncomment the next line
                # raise e

        return scores

    return get_reward_from_environment

# Example of how to use it in your notebook:
# get_reward_fn = create_reward_fn(env)
