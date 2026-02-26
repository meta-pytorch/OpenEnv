Codebase rules:
- Prioritize simplicity.
- Do not add traits like Send + Sync unless absolutely necessary.
- Do not add unnecessary inline comments.
- Do not remove pre-existing comments unless necessary.
- Do not add a new readme.md unless necessary.
- Do not write verbose readmes.
- In simulator tests, be extra careful to not introduce additional non-simulator threads.
- Never use fixed simulator seeds unless you require a specific ordering.
- When moving files, use sl mv to preserve history.
- Write tests compactly.
- Be intentional with Rc vs Arc usage.
- When possible, use exhaustive matching.
- Avoid unnecessary RefCells and RwLocks.

sl / hg usage:
- Always run `arc f` before every sl/hg commit or amend.

Modes: run in two modes: Long and Short. Run in short mode if the mode is not specified.

In Long Mode, do the following:
- Before: run tree on the agentbus folder to understand the structure.
- After: run: arc lint
- After: run: arc f
- After: run clippy: cargo clippy --workspace -- -D warnings
- After: check formatting: cargo fmt --check
- After: double check any readme files to make sure they are consistent after your changes.
- For the lint commands, expect them to ask for user input. Answer "y" to all questions.

In Short Mode, run tests after your changes: buck2 test //agentbus/...
