#!/usr/bin/env python3
# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "torch",
#     "transformers>=5.0.0",
#     "peft",
#     "trackio",
#     "openenv-browsergym-env @ git+https://huggingface.co/spaces/openenv/browsergym_env",
# ]
# ///

"""Improve BrowserGym reward with LFM2.5-230M on HF Sandbox."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

import torch
from browsergym_env import BrowserGymAction, BrowserGymEnv
from openenv.core.containers.runtime.hf_sandbox_provider import HFSandboxProvider
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


MODEL_ID = "LiquidAI/LFM2.5-230M"
SANDBOX_IMAGE = "hf.co/spaces/openenv/browsergym_env"
SANDBOX_FLAVOR = "cpu-basic"
BENCHMARK = "miniwob"
TASK_NAME = "click-test"
MAX_NEW_TOKENS = 64
MAX_TRAIN_STEPS = 160
EVAL_EVERY = 20
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
LORA_R = 16
LORA_ALPHA = 32

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an element in the browser by bid.",
            "parameters": {
                "type": "object",
                "properties": {"bid": {"type": "string"}},
                "required": ["bid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "Do nothing in the browser.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

LFM_RESPONSE_SCHEMA = {
    "x-regex": (
        r"^(?P<content>(?:(?!<\|tool_call_start\|>)[\s\S])*?)"
        r"(?P<tool_calls><\|tool_call_start\|>[\s\S]*?<\|tool_call_end\|>)?"
        r"\s*(?:<\|im_end\|>|$)"
    ),
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"([A-Za-z_]\w*\([^)]*\))",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "x-regex": r"^([A-Za-z_]\w*)\(",
                            },
                            "arguments": {
                                "type": "object",
                                "x-regex-key-value": (
                                    r"(?P<key>[A-Za-z_]\w*)="
                                    r"(?P<value>'[^']*'|\"[^\"]*\"|[^,)]*)"
                                ),
                                "additionalProperties": {
                                    "type": "string",
                                    "x-regex": r"^['\"]?(.*?)['\"]?$",
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

SYSTEM_PROMPT = """You control a web browser to complete tasks.
The page structure shows elements as: [bid] element_type 'element_text'.
When the task asks you to click something, call the click tool with the matching bid."""


@dataclass
class EvalResult:
    reward: float
    done: bool
    action: str | None
    completion: str
    parsed: dict[str, Any] | None
    goal: str
    axtree: str


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


def setup_trackio(timestamp: str):
    try:
        import trackio

        trackio.init(
            space_id=f"trackio-browsergym-lfm25-230m-{timestamp}",
            project="openenv-browsergym-hf-sandbox",
            name="lfm25-230m-click-test-reward-improvement",
            config={
                "model_id": MODEL_ID,
                "task_name": TASK_NAME,
                "learning_rate": LEARNING_RATE,
                "max_train_steps": MAX_TRAIN_STEPS,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
            },
        )
        return trackio
    except Exception as exc:
        print(f"Trackio disabled: {exc!r}")
        return None


def load_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        print(f"AutoTokenizer failed, using fast tokenizer fallback: {exc!r}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            MODEL_ID,
            extra_special_tokens={},
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.response_schema = LFM_RESPONSE_SCHEMA
    return tokenizer


def make_provider() -> HFSandboxProvider:
    return HFSandboxProvider(
        image=SANDBOX_IMAGE,
        flavor=SANDBOX_FLAVOR,
        env_vars={
            "BROWSERGYM_BENCHMARK": BENCHMARK,
            "BROWSERGYM_TASK_NAME": TASK_NAME,
            "BROWSERGYM_HEADLESS": "true",
            "BROWSERGYM_VIEWPORT_WIDTH": "332",
            "BROWSERGYM_VIEWPORT_HEIGHT": "214",
            "MINIWOB_URL": "file:///app/miniwob-plusplus/miniwob/html/miniwob/",
        },
    )


def format_observation(goal: str, axtree: str) -> str:
    return f"Goal: {goal}\n\nPage structure:\n{axtree}"


def build_messages(goal: str, axtree: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(goal, axtree)},
    ]


def build_prompt(tokenizer, goal: str, axtree: str) -> str:
    return tokenizer.apply_chat_template(
        build_messages(goal, axtree),
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_supervised_example(
    tokenizer,
    goal: str,
    axtree: str,
    bid: str,
) -> tuple[str, str]:
    messages = build_messages(goal, axtree)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )
    target_messages = messages + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "click", "arguments": {"bid": bid}},
                }
            ],
        }
    ]
    full_text = tokenizer.apply_chat_template(
        target_messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=False,
    )
    if full_text.startswith(prompt_text):
        return prompt_text, full_text
    return (
        prompt_text,
        f"{prompt_text}<|tool_call_start|>[click(bid='{bid}')]"
        "<|tool_call_end|><|im_end|>",
    )


def extract_first_button_bid(axtree: str) -> str:
    bids = re.findall(r"\[(\d+)\]\s+button", axtree)
    if not bids:
        raise RuntimeError(f"No button bid found in axtree: {axtree!r}")
    return bids[0]


def parse_action(
    tokenizer, completion: str
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = tokenizer.parse_response(completion)
    except Exception as exc:
        print(f"Failed to parse completion: {exc!r}\n{completion!r}")
        return None, None

    tool_calls = parsed.get("tool_calls") or []
    if not tool_calls:
        return parsed, None

    function = tool_calls[0].get("function", {})
    name = function.get("name")
    arguments = function.get("arguments") or {}
    if name == "click" and arguments.get("bid") is not None:
        return parsed, f"click({str(arguments['bid'])!r})"
    if name == "noop":
        return parsed, "noop()"
    return parsed, None


def evaluate(model, tokenizer, env, device: torch.device) -> EvalResult:
    model.eval()
    reset = env.reset(task_name=TASK_NAME)
    observation = reset.observation
    goal = observation.goal or ""
    axtree = observation.axtree_txt or ""
    prompt = build_prompt(tokenizer, goal, axtree)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        output[0, inputs["input_ids"].shape[-1] :],
        skip_special_tokens=False,
    )
    parsed, action = parse_action(tokenizer, completion)
    reward = 0.0
    done = False
    if action is not None:
        step = env.step(BrowserGymAction(action_str=action))
        reward = float(step.reward or 0.0)
        done = bool(step.done)
    return EvalResult(
        reward=reward,
        done=done,
        action=action,
        completion=completion,
        parsed=parsed,
        goal=goal,
        axtree=axtree,
    )


def train_step(
    model,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = output.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path("outputs")
        / f"browsergym-lfm25-230m-{sanitize_name(TASK_NAME)}-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trackio = setup_trackio(timestamp)
    tokenizer = load_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.to(device)
    model.print_trainable_parameters()

    provider = make_provider()
    history = []
    final_eval = None

    with BrowserGymEnv(
        message_timeout_s=120.0,
        max_message_size_mb=100.0,
        provider=provider,
    ).sync() as browsergym_env:
        reset = browsergym_env.reset(task_name=TASK_NAME)
        goal = reset.observation.goal or ""
        axtree = reset.observation.axtree_txt or ""
        bid = extract_first_button_bid(axtree)
        oracle = browsergym_env.step(BrowserGymAction(action_str=f"click({bid!r})"))
        print(f"Oracle action click({bid!r}) reward: {oracle.reward}")

        baseline = evaluate(model, tokenizer, browsergym_env, device)
        history.append({"step": 0, "phase": "baseline", **asdict(baseline)})
        print(f"Baseline reward: {baseline.reward}")
        print(f"Baseline completion: {baseline.completion!r}")

        prompt_text, full_text = build_supervised_example(tokenizer, goal, axtree, bid)
        encoded_full = tokenizer(full_text, return_tensors="pt")
        encoded_prompt = tokenizer(prompt_text, return_tensors="pt")
        input_ids_one = encoded_full["input_ids"][0]
        labels_one = input_ids_one.clone()
        labels_one[: encoded_prompt["input_ids"].shape[-1]] = -100
        attention_one = torch.ones_like(input_ids_one)
        input_ids = input_ids_one.unsqueeze(0).repeat(BATCH_SIZE, 1).to(device)
        labels = labels_one.unsqueeze(0).repeat(BATCH_SIZE, 1).to(device)
        attention_mask = attention_one.unsqueeze(0).repeat(BATCH_SIZE, 1).to(device)

        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=LEARNING_RATE,
        )

        improved = False
        last_loss = None
        for step in range(1, MAX_TRAIN_STEPS + 1):
            last_loss = train_step(model, optimizer, input_ids, attention_mask, labels)
            if step == 1 or step % 10 == 0:
                print(f"Step {step}: loss={last_loss:.6f}")
            if trackio is not None:
                try:
                    trackio.log({"loss": last_loss, "train_step": step})
                except Exception as exc:
                    print(f"Trackio logging disabled: {exc!r}")
                    trackio = None
            if step % EVAL_EVERY == 0 or step == MAX_TRAIN_STEPS:
                current = evaluate(model, tokenizer, browsergym_env, device)
                final_eval = current
                record = {
                    "step": step,
                    "phase": "eval",
                    "loss": last_loss,
                    **asdict(current),
                }
                history.append(record)
                print(json.dumps(record, indent=2))
                if current.reward > baseline.reward:
                    improved = True
                    print(
                        "Reward improved: "
                        f"{baseline.reward} -> {current.reward} at step {step}"
                    )
                    break

        if not improved:
            print(
                "Reward did not improve: "
                f"baseline={baseline.reward}, final={final_eval}"
            )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "task_name": TASK_NAME,
                "baseline": asdict(baseline),
                "final_eval": asdict(final_eval) if final_eval else None,
                "history": history,
                "parameters": {
                    "max_train_steps": MAX_TRAIN_STEPS,
                    "eval_every": EVAL_EVERY,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "sandbox_flavor": SANDBOX_FLAVOR,
                },
            },
            indent=2,
        )
    )
    print(f"Saved adapter and run summary to {output_dir}")


if __name__ == "__main__":
    main()
