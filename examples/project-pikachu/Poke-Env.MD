### Poke-Env Environment:

The Poke-Env is a Python library that provides an environment for training reinforcement learning agents to play Pokémon battles. It simulates the mechanics of Pokémon games, allowing agents to learn strategies and make decisions in battles. Gen8OU is the game format used in this example.

#### Installation:
```bash
pip install poke-env
```

#### Basic Usage:
Poke-Env allows battles against a random player or bots. Each battle consists of turns where the agent selects actions based on the current state of the game.

```json
 {"action_type": "move" or "switch"}
```

Each battle outputs the observation of what the team or player has access to

team_1 = """
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
Dragon Pulse
Flamethrower
Sludge Wave
Thunderbolt

Sylveon (M) @ Leftovers
Ability: Pixilate
EVs: 248 HP / 244 Def / 16 SpD
Calm Nature
IVs: 0 Atk
Hyper Voice
Mystical Fire
Protect
Wish

Toxtricity (M) @ Throat Spray
Ability: Punk Rock
EVs: 4 Atk / 252 SpA / 252 Spe
Rash Nature
Overdrive
Boomburst
Shift Gear
Fire Punch

Seismitoad (M) @ Leftovers
Ability: Water Absorb
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
Stealth Rock
Scald
Earthquake
Toxic

Corviknight (M) @ Leftovers
Ability: Pressure
EVs: 248 HP / 80 SpD / 180 Spe
Impish Nature
Defog
Brave Bird
Roost
U-turn

Galvantula @ Focus Sash
Ability: Compound Eyes
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
Sticky Web
Thunder Wave
Thunder
Energy Ball
"""

plus what player or team 2 has 

team_2 = """
Togekiss @ Leftovers
Ability: Serene Grace
EVs: 248 HP / 8 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
Air Slash
Nasty Plot
Substitute
Thunder Wave

Galvantula @ Focus Sash
Ability: Compound Eyes
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
Sticky Web
Thunder Wave
Thunder
Energy Ball

Cloyster @ Leftovers
Ability: Skill Link
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
Icicle Spear
Rock Blast
Ice Shard
Shell Smash

Sandaconda @ Focus Sash
Ability: Sand Spit
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
Stealth Rock
Glare
Earthquake
Rock Tomb

Excadrill @ Focus Sash
Ability: Sand Rush
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
Iron Head
Rock Slide
Earthquake
Rapid Spin

Cinccino @ Leftovers
Ability: Skill Link
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
Bullet Seed
Knock Off
Rock Blast
Tail Slap
"""
