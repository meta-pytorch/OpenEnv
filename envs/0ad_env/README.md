openenv-0ad-bridge is a high-fidelity reinforcement learning environment built on the 0 A.D. engine - the premier open source ancient warfare RTS. It bridges the gap between complex game logic and AI agents via a standardized OpenEnv interface.

<!-- **0 A.D.** itself is too large and complex to distribute as a lightweight, HuggingFace-style environment, and must be installed separately. For that reason, this repository focuses on the integration layer and documentation rather than bundling the game itself. Feedback on better ways to package or ship this integration is welcome. 
Repository: [https://github.com/0xrushi/openenv-0ad-bridge](https://github.com/0xrushi/openenv-0ad-bridge)
-->

## Key Features

- **Deep RL Ready**: Native support for high-dimensional observation spaces and complex action hierarchies.
- **Open-Source Core**: Leverages the robust, cross-platform engine of 0 A.D. (Alpha 26+).
- **Flexible Integration**: Includes a dedicated C++/Python integration layer for low-latency state sampling.

## Installation & Requirements

Note: This repository contains the integration layer and documentation. A local installation of the 0 A.D. game client is required.

### Prerequisites

- **Docker** (with **Docker Compose**)
- **0 A.D. game client** installed locally

## Demo video

Video demo of agent thinking: [https://www.youtube.com/watch?v=qCLUHdn229w&feature=youtu.be](https://www.youtube.com/watch?v=qCLUHdn229w&feature=youtu.be)

## Getting Started: Gemini agent test

This environment is powered by the external bridge repo (`openenv-0ad-bridge`). To run a minimal “hello world” Gemini agent test:

1. Clone the bridge repo:

```bash
git clone https://github.com/0xrushi/openenv-0ad-bridge
cd openenv-0ad-bridge
```

2. Set the **0 A.D. binary path** (either edit `launcher.py`, or plan to pass `--binary` / set `ZEROAD_BINARY` when launching).
3. Export your Gemini API key:

```bash
export GEMINI_API_KEY=your-key-here
```

4. Start the dockerized stack:

```bash
docker compose up -d
```

5. Run the multi-provider match using the Gemini config:

```bash
python tools/multi_provider_match.py --config configs/gemini_vs_ai.toml
```

## Contributing

- **New agent PRs are welcome**.
- **Please raise issues for new API interfaces** you’d like to see added.

## Limitations & Future Work

* Requires a local **0 A.D.** installation; cannot run in a fully hosted environment or simple notebook
* Command processing and higher-level action abstractions are still under development
* No built-in reward specification yet; current focus is on environment control and experimentation rather than training-ready reward shaping
* Additional example agents, prompts, and evaluation setups are needed
