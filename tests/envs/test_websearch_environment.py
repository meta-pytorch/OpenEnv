import os
from envs.websearch_env.server import WebSearchEnvironment
from envs.websearch_env.models import WebSearchAction, WebSearchObservation

def test_websearch_environment():

    # Check if the SERPER_API_KEY is set
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        import pytest
        pytest.skip("Skipping websearch environment test because SERPER_API_KEY is not set.")

    # Create the environment
    env = WebSearchEnvironment()

    # Reset the environment
    obs: WebSearchObservation = env.reset()
    assert obs.web_contents == []
    assert obs.content == ""

    # Step the environment
    obs: WebSearchObservation = env.step(WebSearchAction(query="What is the capital of France?"))
    if not obs.metadata.get("error"):
        assert obs.web_contents != []
        assert len(obs.web_contents) == 5
        assert obs.metadata == {"query": "What is the capital of France?"}
    else:
        assert obs.web_contents == []
        assert "[ERROR]" in obs.content