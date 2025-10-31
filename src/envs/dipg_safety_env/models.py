# src/envs/dipg_safety_env/models.py

from dataclasses import dataclass, field
from core.env_server import Action, Observation, State

@dataclass
class DIPGAction(Action):
    """The action taken by the agent, which is its generated response."""
    llm_response: str

@dataclass
class DIPGObservation(Observation):
    """The observation given to the agent: a context and a question."""
    context: str
    question: str

@dataclass
class DIPGState(State):
    """The internal state of the environment for tracking the current challenge."""
    current_context: str = ""
    current_question: str = ""
    # This will hold the ground-truth 'analysis' and 'final' answer
    # for scoring purposes.
    expected_answer: dict = field(default_factory=dict)