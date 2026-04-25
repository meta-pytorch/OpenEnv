"""
Oversight Inbox Arena — Gradio UI
Premium demo interface for the hackathon judges.
"""

import gradio as gr
import random

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
    status = f"✅ Queue started! {info.get('queue_size', '?')} tickets in {difficulty.upper()} mode. Seed: {seed}"
    return env, obs, info, ticket_md, spec_md, stats_md, status, 0.0, ""


def do_step(env, obs, category, priority, escalate):
    if env is None:
        return env, obs, {}, "—", "—", "—", "⚠️ Please click **Start Queue** first!", 0.0, ""

    action = EmailTriageAction(
        category=category,
        priority=int(priority),
        should_escalate=bool(escalate),
    )
    obs = env.step(action)
    info = obs.info or {}
    comps = info.get("reward_components", {})

    ticket_md = _fmt_ticket(obs) if not obs.done else "### 🎉 Queue Complete!\nAll tickets have been processed."
    spec_md = _fmt_specialists(info) if not obs.done else ""
    stats_md = _fmt_stats(info)

    reward_breakdown = ""
    if comps:
        reward_breakdown = (
            f"Quality: **{comps.get('quality', 0):.2f}** | "
            f"SLA: **{comps.get('sla', 0):.2f}** | "
            f"Policy: **{comps.get('policy', 0):.2f}** | "
            f"Oversight: **{comps.get('oversight', 0):.2f}**"
        )

    if obs.done:
        s = env._state
        status = (
            f"🏁 Episode finished! Resolved {s.tickets_resolved}/{s.queue_size} tickets. "
            f"Total reward: **{s.total_reward:.3f}**"
        )
    else:
        remaining = info.get("tickets_remaining", "?")
        drift = " ⚠️ SCHEMA DRIFT ACTIVE!" if info.get("policy_drift_occurred") else ""
        status = f"Step submitted. Reward: **{obs.reward:.3f}** | {remaining} tickets remaining.{drift}"

    return env, obs, info, ticket_md, spec_md, stats_md, status, float(obs.reward), reward_breakdown


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_ticket(obs) -> str:
    if obs is None:
        return "_No ticket loaded. Click **Start Queue** to begin._"
    d = obs if isinstance(obs, dict) else obs.model_dump() if hasattr(obs, "model_dump") else vars(obs)
    internal = "🏢 **INTERNAL**" if d.get("is_internal") else "🌐 **EXTERNAL**"
    return (
        f"**Subject:** {d.get('subject', '—')}\n\n"
        f"**From:** {d.get('sender', '—')} ({d.get('sender_domain', '—')}) {internal}\n\n"
        f"---\n\n{d.get('body_snippet', d.get('body', '—'))}"
    )


def _fmt_specialists(info: dict) -> str:
    reports = info.get("specialist_reports", {})
    if not reports:
        return "_No specialist reports available._"
    lines = []
    icons = {"triage": "🔵", "compliance": "🟡", "priority": "🔴", "routing": "🟢"}
    for name, data in reports.items():
        icon = icons.get(name, "⚪")
        lines.append(f"**{icon} {name.title()} Specialist**")
        if "category" in data:
            lines.append(f"- Suggests category: `{data['category']}`")
        if "priority" in data:
            lines.append(f"- Suggests priority: `{data['priority']}`")
        if "recommended_action" in data:
            lines.append(f"- Action: `{data['recommended_action']}`")
        conf = data.get("confidence", None)
        if conf is not None:
            bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            lines.append(f"- Confidence: `{bar}` {conf:.0%}")
        if "flagged" in data and data["flagged"]:
            lines.append(f"- ⚠️ **FLAGGED**: {data.get('reason', 'policy issue')}")
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
        f"**SLA Breaches:** {sla} &nbsp;&nbsp; **Policy Violations:** {pol}",
        f"**Oversight Catches:** {catches} &nbsp;&nbsp; **Drift Events:** {drift}",
    ]
    if drift > 0:
        lines.append("⚠️ **SCHEMA DRIFT ACTIVE — Rules have changed mid-shift!**")
    return "  \n".join(lines)


# ── UI builder ────────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
.ticket-box { background: #1e1e2e; border: 1px solid #4a4a6a; border-radius: 12px; padding: 16px; }
.spec-box   { background: #12121f; border: 1px solid #3a3a5a; border-radius: 12px; padding: 16px; }
.stats-bar  { background: #16213e; border-radius: 8px; padding: 10px 16px; font-size: 0.9em; }
.reward-pill{ background: #7c3aed; color: white; border-radius: 20px; padding: 4px 14px; font-weight: bold; }
footer { display: none !important; }
"""

INTRO_MD = """
# 🛡️ Oversight Inbox Arena
### Multi-Agent Reinforcement Learning Environment — Grand Finale Demo

> **You are the Coordinator AI.** Your team of 4 specialist agents has already reviewed each email.  
> Read their advice, then make the final call: **category · priority · escalate?**  
> Rules can change mid-shift (Schema Drift). Catch specialist mistakes. Don't let SLA timers expire.

**Reward signals (5 independent):** Quality · SLA · Policy Compliance · Oversight · Anti-Hacking
"""

HOWTO_MD = """
### How to Play
1. Pick a **difficulty** and click **🚀 Start Queue**
2. Read the email ticket on the left
3. Check what your specialists recommend on the right  
4. Choose your **Category**, **Priority** and whether to **Escalate**
5. Click **✅ Submit Decision** — get instant reward feedback
6. Repeat until the queue is cleared!

*Hard & Adversarial modes include schema drift — watch for rule-change warnings!*
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Oversight Inbox Arena", theme=THEME, css=CSS) as demo:

        # ── Shared state ─────────────────────────────────────────────────────
        env_s  = gr.State(None)
        obs_s  = gr.State(None)
        info_s = gr.State({})

        # ── Hero section ─────────────────────────────────────────────────────
        gr.Markdown(INTRO_MD)

        with gr.Accordion("📖 How to Play", open=False):
            gr.Markdown(HOWTO_MD)

        gr.Markdown("---")

        # ── Controls row ─────────────────────────────────────────────────────
        with gr.Row():
            difficulty = gr.Dropdown(
                choices=["easy", "medium", "hard", "adversarial"],
                value="hard",
                label="🎯 Difficulty",
                scale=1,
            )
            start_btn = gr.Button("🚀 Start Queue", variant="primary", scale=1)
            status_md = gr.Markdown("_Click **Start Queue** to begin a new episode._", scale=4)

        # ── Stats bar ────────────────────────────────────────────────────────
        stats_md = gr.Markdown("", elem_classes=["stats-bar"])

        gr.Markdown("---")

        # ── Main arena ───────────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown("### 📨 Incoming Email")
                ticket_md = gr.Markdown(
                    "_No ticket yet. Start the queue above._",
                    elem_classes=["ticket-box"],
                )

            with gr.Column(scale=4):
                gr.Markdown("### 🤖 Specialist Panel")
                spec_md = gr.Markdown(
                    "_Specialists will report here once the queue starts._",
                    elem_classes=["spec-box"],
                )

        gr.Markdown("---")

        # ── Decision row ─────────────────────────────────────────────────────
        gr.Markdown("### ⚡ Your Coordinator Decision")
        with gr.Row():
            cat_in  = gr.Dropdown(
                choices=["billing", "support", "spam", "urgent", "marketing", "other"],
                value="support",
                label="📂 Category",
                scale=2,
            )
            pri_in  = gr.Slider(minimum=1, maximum=5, step=1, value=3,
                                label="🔢 Priority  (1=Low · 5=Critical)", scale=3)
            esc_in  = gr.Checkbox(label="🚨 Escalate to Human Reviewer?", scale=1)
            sub_btn = gr.Button("✅ Submit Decision", variant="secondary", scale=1)

        # ── Reward row ───────────────────────────────────────────────────────
        with gr.Row():
            reward_num = gr.Number(label="⭐ Step Reward", value=0.0, precision=3, scale=1)
            reward_breakdown = gr.Markdown("", scale=5)

        gr.Markdown("---")
        gr.Markdown(
            "_Built with 🤗 Hugging Face · Unsloth · PyTorch · GRPO · FastAPI_  \n"
            "_[GitHub](https://github.com/Rhushya/OpenEnv) · Meta PyTorch OpenEnv Hackathon 2025_"
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
    app = build_ui()
    app.launch(share=True)
