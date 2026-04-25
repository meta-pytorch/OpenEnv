from __future__ import annotations

"""
Oversight Inbox Arena — Gradio UI
Clean black-and-white demo interface.
"""

import random
from typing import Any

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    from envs.email_triage_env.server.email_triage_environment import EmailTriageEnvironment
    from envs.email_triage_env.models import EmailTriageAction
except ImportError:
    try:
        from email_triage_env.server.email_triage_environment import EmailTriageEnvironment
        from email_triage_env.models import EmailTriageAction
    except ImportError:
        from server.email_triage_environment import EmailTriageEnvironment
        from models import EmailTriageAction


# ── Environment helpers ───────────────────────────────────────────────────────

def do_reset(difficulty):
    seed = random.randint(0, 9999)
    env = EmailTriageEnvironment(difficulty=difficulty)
    obs = env.reset(seed=seed, difficulty=difficulty)
    info = obs.info or {}

    ticket_md = _fmt_ticket(obs)
    spec_md = _fmt_specialists(info)
    stats_md = _fmt_stats(info)
    status = f"Queue started — {info.get('queue_size', '?')} tickets in {difficulty.upper()} mode  |  Seed: {seed}"
    return env, obs, info, ticket_md, spec_md, stats_md, status, 0.0, ""


def do_step(env, obs, category, priority, escalate):
    if env is None:
        return env, obs, {}, "—", "—", "—", "Click **Start Queue** first.", 0.0, ""

    action = EmailTriageAction(
        category=category,
        priority=int(priority),
        should_escalate=bool(escalate),
    )
    obs = env.step(action)
    info = obs.info or {}
    comps = info.get("reward_components", {})

    ticket_md = _fmt_ticket(obs) if not obs.done else "### Queue Complete\nAll tickets have been processed."
    spec_md = _fmt_specialists(info) if not obs.done else ""
    stats_md = _fmt_stats(info)

    reward_breakdown = ""
    if comps:
        reward_breakdown = (
            f"Quality: **{comps.get('quality', 0):.2f}**  |  "
            f"SLA: **{comps.get('sla', 0):.2f}**  |  "
            f"Policy: **{comps.get('policy', 0):.2f}**  |  "
            f"Oversight: **{comps.get('oversight', 0):.2f}**"
        )

    if obs.done:
        s = env.state
        status = (
            f"Episode finished — Resolved {s.tickets_resolved}/{s.queue_size} tickets  |  "
            f"Total reward: {s.total_reward:.3f}"
        )
    else:
        remaining = info.get("tickets_remaining", "?")
        drift = "  ⚠ SCHEMA DRIFT ACTIVE" if info.get("policy_drift_occurred") else ""
        status = f"Step submitted  |  Reward: {obs.reward:.3f}  |  {remaining} tickets remaining{drift}"

    return env, obs, info, ticket_md, spec_md, stats_md, status, float(obs.reward), reward_breakdown


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_ticket(obs) -> str:
    if obs is None:
        return "_No ticket loaded. Click **Start Queue** to begin._"
    d = obs if isinstance(obs, dict) else obs.model_dump() if hasattr(obs, "model_dump") else vars(obs)
    internal = "INTERNAL" if d.get("is_internal") else "EXTERNAL"
    return (
        f"**Subject:** {d.get('subject', '—')}\n\n"
        f"**From:** {d.get('sender', '—')} ({d.get('sender_domain', '—')})  [{internal}]\n\n"
        f"---\n\n{d.get('body_snippet', d.get('body', '—'))}"
    )


def _fmt_specialists(info: dict) -> str:
    reports = info.get("specialist_reports", {})
    if not reports:
        return "_No specialist reports available._"
    lines = []
    labels = {"triage": "Triage", "compliance": "Compliance", "priority": "Priority", "routing": "Routing"}
    for name, data in reports.items():
        label = labels.get(name, name.title())
        lines.append(f"**{label} Specialist**")
        if "category" in data:
            lines.append(f"- Category: `{data['category']}`")
        if "priority" in data:
            lines.append(f"- Priority: `{data['priority']}`")
        if "recommended_action" in data:
            lines.append(f"- Action: `{data['recommended_action']}`")
        conf = data.get("confidence", None)
        if conf is not None:
            bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            lines.append(f"- Confidence: `{bar}` {conf:.0%}")
        if data.get("flagged"):
            lines.append(f"- ⚠ FLAGGED: {data.get('reason', 'policy issue')}")
        lines.append("")
    return "\n".join(lines)


def _fmt_stats(info: dict) -> str:
    state = info.get("state", {})
    if not state:
        return ""
    resolved = state.get("tickets_resolved", 0)
    total = state.get("queue_size", 0)
    sla = state.get("sla_breaches", 0)
    pol = state.get("policy_violations", 0)
    drift = state.get("drift_count", 0)
    catches = state.get("oversight_catches", 0)
    pct = int((resolved / total * 100)) if total else 0
    bar = "█" * (pct // 10) + "░" * (10 - pct // 10)

    lines = [
        f"**Progress:** `{bar}` {resolved}/{total} ({pct}%)",
        f"**SLA Breaches:** {sla}    **Policy Violations:** {pol}",
        f"**Oversight Catches:** {catches}    **Drift Events:** {drift}",
    ]
    if drift > 0:
        lines.append("⚠ SCHEMA DRIFT ACTIVE — Rules have changed mid-shift!")
    return "  \n".join(lines)


# ── UI builder ────────────────────────────────────────────────────────────────

CSS = """
/* ── FORCE WHITE EVERYWHERE ── */
body,
.gradio-container,
.gr-block, .gr-box, .gr-form, .gr-panel, .gr-group,
.gr-padded, .gr-compact,
.contain,
div[class*="block"], div[class*="wrap"],
.dark, [data-testid] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #111111 !important;
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
}

/* ── Every Gradio column, row, tab, accordion ── */
.gr-row, .gr-column, .gr-tab, .gr-tabs,
.gr-accordion, .gr-accordion-header,
.svelte-1drgfvp, .svelte-1gfkn6j,
div[class*="container"], div[class*="column"],
div[class*="row"], div[class*="panel"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* ── Markdown containers ── */
.gr-markdown, .gr-markdown p, .gr-markdown li,
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3,
.gr-markdown h4, .gr-markdown strong, .gr-markdown em,
.prose, .prose p, .prose li, .prose h1, .prose h2, .prose h3 {
    color: #000000 !important;
    background: transparent !important;
}

/* ── Links ── */
.gr-markdown a, .prose a { color: #000000 !important; text-decoration: underline; }

/* ── Header ── */
.arena-header {
    border-bottom: 2px solid #111111;
    padding: 24px 0 16px 0;
    margin-bottom: 8px;
    background: #ffffff !important;
}
.arena-title {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #000000 !important;
    margin: 0 0 4px 0;
}
.arena-subtitle {
    font-size: 0.875rem;
    color: #000000 !important;
    margin: 0;
}

/* ── Ticket & Specialist panels ── */
.panel-ticket, .panel-specialists {
    border: 1px solid #000000 !important;
    border-radius: 6px;
    padding: 16px 20px;
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    min-height: 200px;
}

/* ── Stats bar ── */
.panel-stats {
    border: 1px solid #000000 !important;
    border-radius: 6px;
    padding: 12px 16px;
    background: #ffffff !important;
    background-color: #ffffff !important;
    font-size: 0.875rem;
    color: #000000 !important;
}

/* ── Status bar ── */
.status-bar {
    border-left: 3px solid #000000;
    padding: 8px 12px;
    background: #ffffff !important;
    background-color: #ffffff !important;
    font-size: 0.875rem;
    color: #000000 !important;
    border-radius: 0 4px 4px 0;
}

/* ── Buttons ── */
button.primary, button[class*="primary"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1.5px solid #000000 !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}
button.primary:hover, button[class*="primary"]:hover {
    background: #f0f0f0 !important;
    background-color: #f0f0f0 !important;
}
button.secondary, button[class*="secondary"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1.5px solid #000000 !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}
button.secondary:hover, button[class*="secondary"]:hover {
    background: #f0f0f0 !important;
    background-color: #f0f0f0 !important;
}

/* ── Inputs, dropdowns, sliders ── */
input, select, textarea,
.gr-dropdown, .gr-slider,
div[class*="input"], div[class*="dropdown"],
div[class*="select"], div[class*="slider"] {
    border: 1px solid #000000 !important;
    border-radius: 4px !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #000000 !important;
    outline: none !important;
}

/* ── Labels ── */
label, .gr-label, span[data-testid="block-label"] {
    color: #000000 !important;
}

.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #000000 !important;
    margin-bottom: 8px;
}

/* ── Reward strip ── */
.reward-strip {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 0.875rem;
    font-weight: 500;
}

/* ── Dividers ── */
hr { border: none; border-top: 1px solid #eeeeee; margin: 16px 0; }

/* ── Number inputs ── */
.gr-number input { background: #ffffff !important; color: #000000 !important; }

/* ── Checkbox ── */
.gr-checkbox label { color: #000000 !important; }

/* ── Accordion ── */
.gr-accordion { border-color: #000000 !important; }

/* ── Hide Gradio footer ── */
footer { display: none !important; }

/* ── Force max width ── */
.gradio-container { max-width: 1200px !important; margin: auto; }
"""

INTRO_MD = """
<div class="arena-header">
  <div class="arena-title">Oversight Inbox Arena</div>
  <div class="arena-subtitle">
    Multi-agent RL environment — coordinate 4 specialist agents and triage emails safely under schema drift
  </div>
</div>
"""

HOWTO_MD = """
**How to play**

1. Select a difficulty and click **Start Queue**
2. Read the incoming email on the left
3. Check specialist recommendations on the right
4. Set Category, Priority, and Escalation
5. Click **Submit Decision** to get your reward

Hard and Adversarial modes introduce schema drift — watch the status bar for warnings.
"""


def build_ui() -> gr.Blocks:
    if gr is None:
        raise ImportError("gradio is required to build the UI")

    with gr.Blocks(
        title="Oversight Inbox Arena",
        theme=gr.themes.Base(
            primary_hue="neutral",
            secondary_hue="neutral",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        ),
        css=CSS,
    ) as demo:

        # ── Shared state ─────────────────────────────────────────────────────
        env_s  = gr.State(None)
        obs_s  = gr.State(None)
        info_s = gr.State({})

        # ── Header ───────────────────────────────────────────────────────────
        gr.Markdown(INTRO_MD)

        with gr.Accordion("How to play", open=False):
            gr.Markdown(HOWTO_MD)

        gr.Markdown("---")

        # ── Controls row ─────────────────────────────────────────────────────
        with gr.Row():
            difficulty = gr.Dropdown(
                choices=["easy", "medium", "hard", "adversarial"],
                value="hard",
                label="Difficulty",
                scale=1,
            )
            start_btn = gr.Button("Start Queue", variant="primary", scale=1)
            status_md = gr.Markdown(
                "_Select a difficulty and click **Start Queue** to begin._",
                elem_classes=["status-bar"],
            )

        # ── Stats bar ────────────────────────────────────────────────────────
        stats_md = gr.Markdown("", elem_classes=["panel-stats"])

        gr.Markdown("---")

        # ── Main arena ───────────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown("**Incoming Email**", elem_classes=["section-label"])
                ticket_md = gr.Markdown(
                    "_No ticket yet. Start the queue above._",
                    elem_classes=["panel-ticket"],
                )

            with gr.Column(scale=4):
                gr.Markdown("**Specialist Panel**", elem_classes=["section-label"])
                spec_md = gr.Markdown(
                    "_Specialists will report here once the queue starts._",
                    elem_classes=["panel-specialists"],
                )

        gr.Markdown("---")

        # ── Decision row ─────────────────────────────────────────────────────
        gr.Markdown("**Your Decision**", elem_classes=["section-label"])
        with gr.Row():
            cat_in = gr.Dropdown(
                choices=["billing", "support", "spam", "urgent", "marketing", "other"],
                value="support",
                label="Category",
                scale=2,
            )
            pri_in = gr.Slider(
                minimum=1, maximum=5, step=1, value=3,
                label="Priority  (1 = Low · 5 = Critical)",
                scale=3,
            )
            esc_in = gr.Checkbox(label="Escalate to Human Reviewer", scale=1)
            sub_btn = gr.Button("Submit Decision", variant="secondary", scale=1)

        # ── Reward row ───────────────────────────────────────────────────────
        with gr.Row():
            reward_num = gr.Number(
                label="Step Reward", value=0.0, precision=3, scale=1
            )
            reward_breakdown = gr.Markdown("", elem_classes=["reward-strip"])

        gr.Markdown("---")
        gr.Markdown(
            "_Built with Hugging Face · TRL · GRPO · FastAPI_  \n"
            "_[GitHub](https://github.com/Rhushya/OpenEnv)_"
        )

        # ── Wire callbacks ───────────────────────────────────────────────────
        start_btn.click(
            fn=do_reset,
            inputs=[difficulty],
            outputs=[env_s, obs_s, info_s, ticket_md, spec_md, stats_md, status_md, reward_num, reward_breakdown],
        )

        sub_btn.click(
            fn=do_step,
            inputs=[env_s, obs_s, cat_in, pri_in, esc_in],
            outputs=[env_s, obs_s, info_s, ticket_md, spec_md, stats_md, status_md, reward_num, reward_breakdown],
        )

    return demo


if __name__ == "__main__":
    if gr is None:
        raise ImportError("gradio is required to launch the UI")
    app = build_ui()
    app.launch(share=True)
