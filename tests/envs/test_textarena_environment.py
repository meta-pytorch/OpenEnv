import pytest
from textarena_env.models import TextArenaAction, TextArenaMessage
from textarena_env.server.environment import TextArenaEnvironment


def test_convert_messages_coalesces_consecutive_characters():
    env = object.__new__(TextArenaEnvironment)

    raw_messages = [
        (0, "[", "PROMPT"),
        (0, "GAME", "PROMPT"),
        (0, "]", "PROMPT"),
        (1, "A", "MESSAGE"),
        (1, "B", "MESSAGE"),
        (2, "!", "MESSAGE"),
    ]

    converted = env._convert_messages(raw_messages)

    assert converted == [
        TextArenaMessage(sender_id=0, content="[GAME]", category="PROMPT"),
        TextArenaMessage(sender_id=1, content="AB", category="MESSAGE"),
        TextArenaMessage(sender_id=2, content="!", category="MESSAGE"),
    ]


def test_wordle_reset_clears_accumulated_state():
    """Test that resetting Wordle environment clears accumulated observation state.

    This test verifies the workaround for TextArena's LLMObservationWrapper,
    which accumulates observations in self.full_observations across resets.
    """
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(
        env_id="Wordle-v0",
        num_players=1,
    )

    # First episode
    obs1 = env.reset()
    prompt1_len = len(obs1.prompt)

    # Make a move to accumulate some state
    env.step(TextArenaAction(message="[CRANE]"))

    # Second episode - should NOT accumulate from first episode
    obs2 = env.reset()
    prompt2_len = len(obs2.prompt)

    # Make another move
    env.step(TextArenaAction(message="[STALE]"))

    # Third episode - should NOT accumulate from previous episodes
    obs3 = env.reset()
    prompt3_len = len(obs3.prompt)

    # All prompts should be the same length (no accumulation)
    assert prompt1_len == prompt2_len, (
        f"Episode 2 accumulated state: {prompt1_len} -> {prompt2_len}"
    )
    assert prompt2_len == prompt3_len, (
        f"Episode 3 accumulated state: {prompt2_len} -> {prompt3_len}"
    )

    # Verify the prompts are actually the same content
    assert obs1.prompt == obs2.prompt
    assert obs2.prompt == obs3.prompt


def _secret_word(env: TextArenaEnvironment) -> str:
    """Read the puzzle from TextArena's own game state, not from the prompt.

    Wordle's prompt is static instructions, so hashing the observation would
    report every episode as identical whether or not the seed took effect.
    """
    return env._ta_env.state.game_state["secret_word"]


def test_reset_seed_is_forwarded_to_textarena():
    """A fixed seed must give the same episode, and different seeds different ones."""
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    seeded = []
    for _ in range(3):
        env.reset(seed=1234)
        seeded.append(_secret_word(env))

    env.reset(seed=999)
    other_seed = _secret_word(env)

    assert len(set(seeded)) == 1, f"seed=1234 produced {sorted(set(seeded))}"
    assert other_seed != seeded[0], "a different seed produced the same episode"


def test_reset_seed_is_reproducible_across_instances():
    """The same seed must survive constructing a fresh environment."""
    pytest.importorskip("textarena", reason="textarena not installed")
    first = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    first.reset(seed=1234)

    second = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    second.reset(seed=1234)

    assert _secret_word(first) == _secret_word(second)


def test_reset_without_seed_still_varies():
    """Forwarding the seed must not accidentally pin unseeded episodes."""
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    words = set()
    for _ in range(8):
        env.reset()
        words.add(_secret_word(env))

    assert len(words) > 1, "unseeded resets should not be deterministic"
