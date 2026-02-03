This folder provides an **OpenEnv-compatible RTS environment** built on top of **0 A.D.**, the open-source *Age of Empires–style* real-time strategy game.

**0 A.D.** itself is too large and complex to distribute as a lightweight, HuggingFace-style environment, and must be installed separately. For that reason, this repository focuses on the integration layer and documentation rather than bundling the game itself. Feedback on better ways to package or ship this integration is welcome.

Repository: [https://github.com/0xrushi/openenv-0ad-bridge](https://github.com/0xrushi/openenv-0ad-bridge)

## Limitations & Future Work

* Requires a local **0 A.D.** installation; cannot run in a fully hosted environment or simple notebook
* Command processing and higher-level action abstractions are still under development
* No built-in reward specification yet; current focus is on environment control and experimentation rather than training-ready reward shaping
* Additional example agents, prompts, and evaluation setups are needed
