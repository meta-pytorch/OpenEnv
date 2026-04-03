try:
    from openenv.core.env_server import create_app
except ImportError:
    from core.env_server import create_app

try:
    from envs.email_triage_env.models import EmailTriageAction, EmailTriageObservation
    from envs.email_triage_env.server.email_triage_environment import EmailTriageEnvironment
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation
    from server.email_triage_environment import EmailTriageEnvironment


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
