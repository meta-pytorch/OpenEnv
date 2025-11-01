## Project Pikachu: Solving Pokemon with RL and SFT

Lead: cpich3g, init27
Contributors: osiris, rycerzes, sub_zero5167, Zeus, Jackson, RioT007

Mentor: init27

### Goals:

- [ ] Add Env for Pokemon that interacts well given current structure (avoid vision since we don't have support for that, text is better)
- [ ] Baseline <10B model performance
- [ ] Setup Synthetic SFT Dataset if needed
- [ ] Setup GRPO Trainer


### Task Assignees:

- [ ] cpich3g and init27: Figure out best environment for the use-case

### Milestones:

- [ ] Add Env
- [ ] Baseline on Env
- [ ] Setup Trainer

### Observations:
SubZero:
1) poke_env seems like a popular choice for Pokemon-RL projects, especially those involving Pokémon Showdown battles.
2) Custom environments using earlier Pokemon versions (like Pokemon Red) often use emulators or simplified OpenAI Gym-like environments.
3) Environments typically frame Pokemon states as numeric tensors or feature dictionaries, avoiding raw image input for easier training.
4) Research projects use reward shaping and parallel training to speed learning.
5) Most projects focus on battle-based interactions, therefore no focus on exploration or world interaction.
