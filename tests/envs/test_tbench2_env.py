import os
import sys
import pytest

# Add the project root to the path for envs imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import camel  # noqa: F401
except Exception:
    camel = None

from envs.tbench2_env.models import Tbench2Action
from envs.tbench2_env.server.tbench2_env_environment import Tbench2Environment


@pytest.mark.skipif(camel is None, reason="camel-ai not installed")
@pytest.mark.skipif(
    os.environ.get("TB2_ENABLE_TESTS", "0") != "1",
    reason="TB2_ENABLE_TESTS not enabled",
)
def test_tbench2_env_smoke():
    env = Tbench2Environment(tasks_dir=os.environ.get("TB2_TASKS_DIR"))
    obs = env.reset(task_id=os.environ.get("TB2_TASK_ID", "headless-terminal"))
    assert obs.instruction

    result = env.step(Tbench2Action(action_type="exec", command="pwd"))
    assert result.success
    assert result.output

    env.step(Tbench2Action(action_type="close"))
    env.close()
