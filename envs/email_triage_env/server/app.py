try:
    from openenv.core.env_server import create_app
except ImportError:
    try:
        from openenv_core.env_server import create_app
    except ImportError:
        from core.env_server import create_app

try:
    from envs.email_triage_env.models import EmailTriageAction, EmailTriageObservation
    from envs.email_triage_env.server.email_triage_environment import EmailTriageEnvironment
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation
    from server.email_triage_environment import EmailTriageEnvironment


try:
    app = create_app(
        EmailTriageEnvironment,
        EmailTriageAction,
        EmailTriageObservation,
        env_name="email_triage_env",
    )
except TypeError:
    # Backward-compatible fallback for the minimal local core helper API.
    app = create_app(EmailTriageEnvironment())


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "email_triage_env",
        "health": "/health",
        "docs": "/docs",
    }


try:
    import gradio as gr
    from envs.email_triage_env.server.ui import build_ui
    ui = build_ui()
    app = gr.mount_gradio_app(app, ui, path="/")
except ImportError:
    pass

def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
