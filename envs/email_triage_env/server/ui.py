import gradio as gr
from envs.email_triage_env.server.email_triage_environment import EmailTriageEnvironment
from envs.email_triage_env.models import EmailTriageAction

def reset_env(difficulty):
    env = EmailTriageEnvironment()
    obs = env.reset(difficulty=difficulty)
    info = obs.info or {}
    return env, obs.model_dump(), info, f"Started {difficulty} mode. Queue size: {info.get('queue_size', 1)}"

def step_env(env, category, priority, escalate):
    if env is None:
        return env, None, None, "Please start a queue first.", 0.0
    
    action = EmailTriageAction(
        category=category,
        priority=int(priority),
        should_escalate=escalate
    )
    obs = env.step(action)
    
    msg = f"Action submitted. Reward: {obs.reward:.3f}"
    if obs.done:
        msg += f"\nQueue finished! Total tickets resolved: {env._state.tickets_resolved}"
    
    return env, obs.model_dump(), obs.info or {}, msg, obs.reward

def format_ticket(obs):
    if not obs:
        return "No active ticket."
    return f"""### Subject: {obs.get('subject', '')}
**From:** {obs.get('sender', '')}
**Body:**
{obs.get('body', '')}"""

def format_specialists(obs):
    if not obs or 'specialist_reports' not in obs:
        return "No specialist data available."
    reports = obs['specialist_reports']
    text = ""
    for spec, data in reports.items():
        text += f"#### 🤖 {spec.title()} Specialist\n"
        text += f"- **Action:** {data.get('recommended_action', 'None')}\n"
        text += f"- **Confidence:** {data.get('confidence', 0):.2f}\n"
        if 'reasoning' in data:
            text += f"- **Reasoning:** {data['reasoning']}\n"
        text += "\n"
    return text

def format_state(info):
    if not info:
        return ""
    state = info.get("state", {})
    text = f"**Tickets Handled:** {state.get('tickets_resolved', 0)} | "
    text += f"**SLA Breaches:** {state.get('sla_breaches', 0)} | "
    text += f"**Policy Violations:** {state.get('policy_violations', 0)}"
    if state.get("drift_count", 0) > 0:
        text += f"\n⚠️ **ACTIVE DRIFT MUTATIONS:** {state.get('drift_count')} (Check rules!)"
    return text


def build_ui():
    with gr.Blocks(title="Oversight Inbox Arena", theme=gr.themes.Soft()) as demo:
        env_state = gr.State(None)
        obs_state = gr.State(None)
        info_state = gr.State(None)
        
        gr.Markdown("# 🛡️ Oversight Inbox Arena")
        gr.Markdown("You are the Coordinator AI. Review the incoming tickets, check the specialist advice, and make the final decision. Watch out for schema drift where the rules change mid-shift!")
        
        with gr.Row():
            with gr.Column(scale=1):
                difficulty_dropdown = gr.Dropdown(
                    choices=["easy", "medium", "hard", "adversarial"], 
                    value="hard", 
                    label="Difficulty Level"
                )
                start_btn = gr.Button("🚀 Start New Queue", variant="primary")
                status_text = gr.Textbox(label="Status", interactive=False)
                state_text = gr.Markdown("")
            
            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📨 Current Ticket")
                        ticket_view = gr.Markdown("Click 'Start New Queue' to begin.")
                    
                    with gr.Column():
                        gr.Markdown("### 🧠 Specialist Advice")
                        specialist_view = gr.Markdown("")

        with gr.Row():
            gr.Markdown("### ⚡ Your Decision")
        with gr.Row():
            category_in = gr.Dropdown(
                choices=["sales", "support", "billing", "spam", "urgent"],
                value="support",
                label="Category"
            )
            priority_in = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Priority (1=Low, 5=High)")
            escalate_in = gr.Checkbox(label="Escalate to Human?", value=False)
            
            submit_btn = gr.Button("✅ Submit Action", variant="secondary")
            reward_out = gr.Number(label="Reward Received")

        # Callbacks
        start_btn.click(
            fn=reset_env,
            inputs=[difficulty_dropdown],
            outputs=[env_state, obs_state, info_state, status_text]
        ).then(
            fn=format_ticket, inputs=[obs_state], outputs=[ticket_view]
        ).then(
            fn=format_specialists, inputs=[obs_state], outputs=[specialist_view]
        ).then(
            fn=format_state, inputs=[info_state], outputs=[state_text]
        )
        
        submit_btn.click(
            fn=step_env,
            inputs=[env_state, category_in, priority_in, escalate_in],
            outputs=[env_state, obs_state, info_state, status_text, reward_out]
        ).then(
            fn=format_ticket, inputs=[obs_state], outputs=[ticket_view]
        ).then(
            fn=format_specialists, inputs=[obs_state], outputs=[specialist_view]
        ).then(
            fn=format_state, inputs=[info_state], outputs=[state_text]
        )
        
    return demo
