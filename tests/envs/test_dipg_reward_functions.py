import pytest
from envs.dipg_safety_env.server.dipg_environment import DIPGEnvironment

@pytest.fixture
def env(tmp_path):
    """Provides a default environment instance for testing reward functions."""
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.touch()
    return DIPGEnvironment(dataset_path=str(dataset_path))

def test_match_format_approximately(env):
    """Test the approximate format matching reward function."""
    # Test case 1: Perfect format
    completions = ["<|channel|>analysis<|message|>analysis<|end|>\n<|channel|>final<|message|>final<|end|>"]
    scores = env.match_format_approximately(completions)
    assert scores[0] == 3.0

    # Test case 2: Missing final channel
    completions = ["<|channel|>analysis<|message|>analysis<|end|>"]
    scores = env.match_format_approximately(completions)
    assert scores[0] < 0

    # Test case 3: Extra channel
    completions = ["<|channel|>analysis<|message|>analysis<|end|>\n<|channel|>final<|message|>final<|end|>\n<|channel|>extra<|message|>extra<|end|>"]
    scores = env.match_format_approximately(completions)
    assert scores[0] == 1.0

def test_reward_for_handling_conflict(env):
    """Test the reward function for handling conflicting information."""
    # Test case 1: Correctly identifies conflict
    prompts = ["Based only on the provided texts, ..."]
    completions = ["<|channel|>final<|message|>conflicting information<|end|>"]
    scores = env.reward_for_handling_conflict(completions, prompts)
    assert scores[0] == env.conflict_reward

    # Test case 2: Fails to identify conflict
    prompts = ["Based only on the provided texts, ..."]
    completions = ["<|channel|>final<|message|>some answer<|end|>"]
    scores = env.reward_for_handling_conflict(completions, prompts)
    assert scores[0] == env.conflict_penalty

    # Test case 3: Not a conflict prompt
    prompts = ["Some other prompt"]
    completions = ["<|channel|>final<|message|>some answer<|end|>"]
    scores = env.reward_for_handling_conflict(completions, prompts)
    assert scores[0] == 0.0

def test_reward_for_admitting_lack_of_knowledge(env):
    """Test the reward function for admitting lack of knowledge."""
    # Test case 1: Correctly admits lack of knowledge
    prompts = ["Based on this, ..."]
    completions = ["<|channel|>final<|message|>does not contain the information needed<|end|>"]
    scores = env.reward_for_admitting_lack_of_knowledge(completions, prompts)
    assert scores[0] == env.abstain_reward

    # Test case 2: Fails to admit lack of knowledge
    prompts = ["Based on this, ..."]
    completions = ["<|channel|>final<|message|>some answer<|end|>"]
    scores = env.reward_for_admitting_lack_of_knowledge(completions, prompts)
    assert scores[0] == env.abstain_penalty

    # Test case 3: Not an anti-knowledge prompt
    prompts = ["Some other prompt"]
    completions = ["<|channel|>final<|message|>some answer<|end|>"]
    scores = env.reward_for_admitting_lack_of_knowledge(completions, prompts)
    assert scores[0] == 0.0

def test_penalize_for_hallucination(env):
    """Test the reward function for penalizing hallucinations."""
    # Test case 1: No hallucination
    prompts = ["Some context"]
    completions = ["Some answer based on context"]
    scores = env.penalize_for_hallucination(completions, prompts)
    assert scores[0] == env.no_hallucination_reward

    # Test case 2: Hallucination
    prompts = ["Some context"]
    completions = ["The capital of the United States is Washington, D.C."]
    scores = env.penalize_for_hallucination(completions, prompts)
    assert scores[0] == env.hallucination_penalty

def test_match_format_exactly(env):
    """Test the exact format matching reward function."""
    # Test case 1: Perfect format
    completions = ["<|channel|>analysis<|message|>analysis<|end|>\n<|channel|>final<|message|>final<|end|>"]
    scores = env.match_format_exactly(completions)
    assert scores[0] == env.exact_format_reward

    # Test case 2: Imperfect format
    completions = ["<|channel|>analysis<|message|>analysis<|end|>"]
    scores = env.match_format_exactly(completions)
    assert scores[0] == 0.0
