# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom Gradio tab for TextArena (e.g. Wordle) – renders a Wordle-style grid in an HTML block.

This module is used as gradio_builder when creating the app; the returned Blocks
appear in the "Custom" tab next to the default "Playground" tab.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import gradio as gr

from openenv.core.env_server.types import EnvironmentMetadata


def _wordle_demo_html() -> str:
    """Static Wordle-style grid HTML for the Custom tab (demo only)."""
    return """
<div class="wordle-demo" style="
  font-family: 'Clear Sans', 'Helvetica Neue', Arial, sans-serif;
  max-width: 320px;
  margin: 0 auto;
  padding: 16px;
  text-align: left;
">
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; margin-bottom: 8px;">
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;">C</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;">R</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;">A</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;">N</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;">E</div>
  </div>
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; margin-bottom: 8px;">
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold; background: #538d4e; color: white; border-color: #538d4e;">S</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold; background: #538d4e; color: white; border-color: #538d4e;">T</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold; background: #b59f3b; color: white; border-color: #b59f3b;">O</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold; background: #538d4e; color: white; border-color: #538d4e;">N</div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold; background: #538d4e; color: white; border-color: #538d4e;">E</div>
  </div>
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px;">
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;"></div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;"></div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;"></div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;"></div>
    <div style="width: 100%; aspect-ratio: 1; border: 2px solid #3a3a3c; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: bold;"></div>
  </div>
  <p style="text-align: center; margin-top: 16px; color: #6b6b6b; font-size: 0.9rem;">
    Play in the <strong>Playground</strong> tab: Reset, then Step with guesses like <code>[crane]</code>.
  </p>
</div>
"""


def build_textarena_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """
    Build the Custom tab Blocks for TextArena: Wordle-style HTML block.

    Signature matches the gradio_builder contract (see docs/customizing-web-ui.md).
    In Gradio 6, gr.HTML(value=...) renders HTML; default html_template is "${value}".
    """
    with gr.Blocks(title=f"{title} — Custom") as blocks:
        gr.Markdown(value=f"# Wordle Visualization")
        gr.Markdown(
            value="This tab shows a **Wordle-style** view. Use the **Playground** tab to "
            "Reset and Step with guesses (e.g. `[crane]`, `[stone]`)."
        )
        # Gradio 6: gr.HTML(value=...) renders the string as HTML (html_template default "${value}")
        gr.HTML(value=_wordle_demo_html(), show_label=False)
    return blocks
