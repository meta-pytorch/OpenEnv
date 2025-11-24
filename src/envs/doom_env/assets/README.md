# Doom Environment GIF Generation

This directory contains GIFs showcasing different ViZDoom scenarios.

## Generating GIFs

To generate GIFs for the scenarios:

```bash
# Install dependencies
pip install numpy imageio vizdoom

# Generate GIFs for default scenarios
cd /path/to/OpenEnv/src/envs/doom_env
python generate_gifs.py

# Or generate specific scenarios
python generate_gifs.py --scenario basic deadly_corridor

# Custom settings
python generate_gifs.py --frames 150 --fps 20 --resolution RES_640X480
```

## Generated GIFs

The script will generate the following GIFs:

- `basic.gif` - Basic scenario with simple movement and shooting
- `deadly_corridor.gif` - Navigate a corridor while avoiding/killing monsters
- `defend_the_center.gif` - Defend the center position against enemies
- `health_gathering.gif` - Collect health packs to survive

## Usage in README

These GIFs are embedded in the main README.md file to showcase the environment's capabilities.

## File Sizes

GIFs are optimized for documentation (80 frames @ 12 fps, 320x240 resolution) to keep file sizes reasonable while showing representative gameplay.
