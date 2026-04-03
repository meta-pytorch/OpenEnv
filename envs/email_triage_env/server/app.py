try:
    from openenv.core.env_server import create_app
except ImportError:
    from core.env_server import create_app

from ..models import EmailTriageAction, EmailTriageObservation
from .email_triage_environment import EmailTriageEnvironment


app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="email_triage_env",
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
