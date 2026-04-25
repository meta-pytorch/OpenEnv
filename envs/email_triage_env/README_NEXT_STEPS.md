# Email Triage Env - Next Steps

This guide is focused on your goals:
- demo UI on Hugging Face Spaces
- train RL model on Google Colab Free Tier (T4 GPU)
- use the trained model for your final demo

## 1) Run Local Sanity Checks First

From repo root:

```bash
PYTHONPATH=src:envs uv run pytest tests/envs/test_email_triage_env.py -v
PYTHONPATH=src:envs uv run pytest tests/envs/test_email_triage_http.py -v
```

If both pass, your env and API are in good shape.

## 2) Deploy the UI as a Hugging Face Space

Your UI is in `envs/email_triage_env/server/ui.py`.
For demo reliability, keep UI and API separated:
- Space A (recommended): Gradio demo UI
- Space B (optional): API server for `/reset`, `/step`, `/state`

### Quick UI Space flow

```bash
pip install -U huggingface_hub
hf auth login
hf repo create YOUR_USERNAME/oversight-inbox-ui --type space --space-sdk gradio
```

Then upload these files into the Space repo:
- `envs/email_triage_env/server/ui.py`
- `envs/email_triage_env/server/email_triage_environment.py`
- `envs/email_triage_env/server/graders.py`
- `envs/email_triage_env/server/scenario_generator.py`
- `envs/email_triage_env/server/schema_drift.py`
- `envs/email_triage_env/server/stakeholders.py`
- `envs/email_triage_env/models.py`
- `envs/email_triage_env/server/email_triage_dataset.json`

And include a `requirements.txt` in the Space:

```txt
gradio
fastapi
pydantic
numpy
```

If any missing dependency error appears in Space logs, add it to `requirements.txt` and redeploy.

## 3) Colab Free Tier (T4) Training Plan

Your `train_grpo.py` is already tuned for low VRAM and supports smoke tests.

### Colab steps

1. Runtime -> Change runtime type -> `T4 GPU`
2. Clone repo and install deps:

```bash
!git clone https://github.com/<your-username>/OpenEnv.git
%cd OpenEnv
!pip install -U pip
!pip install trl transformers accelerate datasets torch huggingface_hub
```

3. Run smoke test first:

```bash
!PYTHONPATH=src:envs python envs/email_triage_env/train_grpo.py --smoke
```

4. Run short real train (safe for free tier):

```bash
!PYTHONPATH=src:envs python envs/email_triage_env/train_grpo.py --model Qwen/Qwen2-0.5B --max-steps 50 --dataset-size 64 --output-dir oversight-arena-grpo-t4
```

5. Push checkpoint to Hugging Face Hub:

```bash
!huggingface-cli login
!PYTHONPATH=src:envs python envs/email_triage_env/train_grpo.py --model Qwen/Qwen2-0.5B --max-steps 50 --dataset-size 64 --output-dir oversight-arena-grpo-t4 --push-to-hub --hub-repo YOUR_USERNAME/oversight-arena-grpo-t4
```

## 4) Use Trained Model in Demo

After training:
- keep Space UI running for judges
- optionally add an inference toggle in UI for `baseline` vs `trained model`
- if using HF Inference API, keep token private in Space secrets

Demo script suggestion:
1. Show queue start
2. Show specialist disagreement
3. Show your coordinator final decision
4. Show reward breakdown
5. Show drift event adaptation

## 5) Recommended Free-Tier Defaults

- Model: `Qwen/Qwen2-0.5B`
- Max steps: `30-50`
- Dataset size: `32-64`
- Keep `num_generations` low (already set in script)
- Save often and push checkpoints

## 6) What To Do After This

- Run a second training pass with improved prompts/system instructions
- Compare baseline vs trained rewards on 5 fixed seeds
- Record a short 2-3 minute demo video from your Space
- Freeze your final Space + model versions before submission
