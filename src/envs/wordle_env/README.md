# Wordle Environment

A word guessing game environment where players try to guess a 5-letter word within 6 attempts, receiving feedback on letter correctness and positioning.

## Overview

The Wordle environment implements the classic word guessing game with the following features:
- 5-letter word guessing with 6 attempts maximum
- Color-coded feedback system (green, yellow, gray)
- Letter position tracking
- Comprehensive game state management
- Intelligent reward system

## Quick Start

```python
from envs.wordle_env import WordleAction, WordleEnv

# Create environment from Docker image
client = WordleEnv.from_docker_image("wordle-env:latest")

# Start a new game
result = client.reset()
print(f"Game started! You have {result.observation.max_attempts} attempts.")

# Make a guess
action = WordleAction(guess="CRANE")
result = client.step(action)

print(f"Guess: {result.observation.guess}")
for feedback in result.observation.feedback:
    print(f"{feedback.letter}: {feedback.status.value}")

print(f"Attempt {result.observation.attempt_number}/{result.observation.max_attempts}")
print(f"Game won: {result.observation.game_won}")

# Cleanup
client.close()
```

## Action Specification

### WordleAction
- `guess` (str): The player's 5-letter word guess
  - Must be exactly 5 letters long
  - Must contain only alphabetic characters
  - Case insensitive (automatically converted to uppercase)

## Observation Specification

### WordleObservation
- `guess` (str): The word that was guessed
- `feedback` (List[LetterFeedback]): Feedback for each letter
- `attempt_number` (int): Current attempt number (1-6)
- `max_attempts` (int): Maximum number of attempts (default: 6)
- `game_won` (bool): Whether the game was won
- `game_lost` (bool): Whether the game was lost
- `correct_word` (str, optional): The target word (only revealed when game ends)
- `used_letters` (List[str]): All letters that have been used
- `correct_letters` (List[str]): Letters in correct positions
- `wrong_position_letters` (List[str]): Letters in wrong positions
- `not_in_word_letters` (List[str]): Letters not in the word
- `done` (bool): Whether the game is over
- `reward` (float): Reward for this step
- `metadata` (dict): Additional game information

### LetterFeedback
- `letter` (str): The letter being evaluated
- `status` (LetterStatus): Status of the letter
  - `CORRECT`: Letter is in the correct position (green)
  - `WRONG_POSITION`: Letter is in the word but wrong position (yellow)
  - `NOT_IN_WORD`: Letter is not in the word (gray)
- `position` (int): Position of the letter in the guess (0-4)

## State Specification

### WordleState
- `episode_id` (str): Unique episode identifier
- `step_count` (int): Number of steps taken
- `target_word` (str): The word to be guessed
- `attempt_number` (int): Current attempt number
- `max_attempts` (int): Maximum number of attempts
- `game_won` (bool): Whether the game was won
- `game_lost` (bool): Whether the game was lost
- `guesses` (List[str]): All guesses made so far
- `all_feedback` (List[List[LetterFeedback]]): Feedback for all guesses
- `used_letters` (List[str]): All letters that have been used
- `correct_letters` (List[str]): Letters in correct positions
- `wrong_position_letters` (List[str]): Letters in wrong positions
- `not_in_word_letters` (List[str]): Letters not in the word

## Rewards

The environment uses a sophisticated reward system:
- **Game Won**: 10.0 - (attempt_number - 1) * 0.5 (bonus for fewer attempts)
- **Game Lost**: -5.0 (penalty for losing)
- **Valid Guess**: correct_letters * 1.0 + wrong_position_letters * 0.5
- **Invalid Guess**: -1.0 (penalty for invalid input)

## Game Rules

1. **Word Selection**: Random 5-letter word from a curated word list
2. **Guess Validation**: Must be exactly 5 letters, alphabetic only
3. **Feedback System**:
   - Green (CORRECT): Letter is in the correct position
   - Yellow (WRONG_POSITION): Letter is in the word but wrong position
   - Gray (NOT_IN_WORD): Letter is not in the word
4. **Win Condition**: Guess the word correctly
5. **Lose Condition**: Fail to guess within 6 attempts

## Building and Running

### Build Docker Image

```bash
# Build the base image first (if not already built)
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .

# Build the Wordle environment image
docker build -t wordle-env:latest -f src/envs/wordle_env/server/Dockerfile .
```

### Run Locally

```bash
# Run the server
docker run -p 8000:8000 wordle-env:latest

# Test the environment
curl http://localhost:8000/health
```

## Usage Examples

### Basic Game Loop

```python
from envs.wordle_env import WordleAction, WordleEnv

client = WordleEnv.from_docker_image("wordle-env:latest")

# Start new game
obs = client.reset()

print(f"Starting Wordle game! You have {obs.observation.max_attempts} attempts.")

# Play until game ends
while not obs.done:
    # Get user input (or AI decision)
    guess = input("Enter your guess: ").upper()
    action = WordleAction(guess=guess)
    
    # Make the guess
    obs = client.step(action)
    
    # Display feedback
    print(f"\nAttempt {obs.observation.attempt_number}: {obs.observation.guess}")
    for feedback in obs.observation.feedback:
        status_emoji = {
            "correct": "🟩",
            "wrong_position": "🟨", 
            "not_in_word": "⬜"
        }
        print(f"{feedback.letter} {status_emoji[feedback.status.value]}", end=" ")
    print()
    
    if obs.observation.game_won:
        print(f"🎉 Congratulations! You won in {obs.observation.attempt_number} attempts!")
        print(f"The word was: {obs.observation.correct_word}")
    elif obs.observation.game_lost:
        print(f"😞 Game over! The word was: {obs.observation.correct_word}")

client.close()
```

### Strategy Testing

```python
from envs.wordle_env import WordleAction, WordleEnv

starting_words = ["CRANE", "SLATE", "ADIEU", "AUDIO", "RAISE"]

for word in starting_words:
    client = WordleEnv.from_docker_image("wordle-env:latest")
    obs = client.reset()
    
    # Make the starting guess
    action = WordleAction(guess=word)
    obs = client.step(action)
    
    print(f"Starting with '{word}':")
    print(f"  Correct letters: {obs.observation.correct_letters}")
    print(f"  Wrong position: {obs.observation.wrong_position_letters}")
    print(f"  Not in word: {obs.observation.not_in_word_letters}")
    
    client.close()
```

### Advanced Game Analysis

```python
# Analyze game patterns

from envs.wordle_env import WordleAction, WordleEnv

def analyze_game():
    client = WordleEnv.from_docker_image("wordle-env:latest")
    obs = client.reset()
    
    game_history = []
    
    while not obs.done:
        # Simple strategy: use common letters
        if obs.observation.attempt_number == 1:
            guess = "CRANE"
        elif obs.observation.attempt_number == 2:
            guess = "SLATE"
        else:
            # Use available letters
            available_letters = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 
                               if c not in obs.observation.not_in_word_letters]
            guess = "".join(available_letters[:5]).ljust(5, 'A')
        
        action = WordleAction(guess=guess)
        obs = client.step(action)
        
        game_history.append({
            'attempt': obs.observation.attempt_number,
            'guess': obs.observation.guess,
            'feedback': [f.status.value for f in obs.observation.feedback],
            'reward': obs.reward
        })
    
    print("Game Analysis:")
    for turn in game_history:
        print(f"  Turn {turn['attempt']}: {turn['guess']} -> {turn['feedback']} (reward: {turn['reward']})")
    
    print(f"Final result: {'Won' if obs.observation.game_won else 'Lost'}")
    if obs.observation.correct_word:
        print(f"Target word: {obs.observation.correct_word}")
    
    client.close()

analyze_game()
```

## Advanced Features

### Custom Game Configuration

```python
# Create environment with custom settings
from envs.wordle_env.server.wordle_environment import WordleEnvironment

env = WordleEnvironment(max_attempts=8)  # 8 attempts instead of 6
```

### State Inspection

```python
# Get detailed game state
state = client.state()
print(f"Episode ID: {state.episode_id}")
print(f"Target word: {state.target_word}")
print(f"Attempts made: {state.attempt_number}")
print(f"All guesses: {state.guesses}")
print(f"Game won: {state.game_won}")
```

### Letter Tracking

```python
# Track letter usage patterns
obs = client.step(WordleAction(guess="CRANE"))

print("Letter Analysis:")
print(f"  Used letters: {obs.observation.used_letters}")
print(f"  Correct positions: {obs.observation.correct_letters}")
print(f"  Wrong positions: {obs.observation.wrong_position_letters}")
print(f"  Not in word: {obs.observation.not_in_word_letters}")
```

## Implementation Details

- **Word List**: Curated list of common 5-letter words
- **Feedback Algorithm**: Proper handling of duplicate letters
- **Reward Structure**: Progressive rewards based on correctness
- **State Tracking**: Complete game history and letter analysis
- **Validation**: Input validation for guesses
- **HTTP Interface**: RESTful API for remote access

## File Structure

```
src/envs/wordle_env/
├── __init__.py              # Export classes
├── models.py                # Action, Observation, State definitions
├── client.py                # HTTP client implementation
├── README.md                # This documentation
└── server/
    ├── __init__.py
    ├── wordle_environment.py # Environment logic
    ├── app.py               # FastAPI application
    └── Dockerfile           # Docker image definition
```

## Word List

The environment includes a curated list of common 5-letter words including:
- CRANE, SLATE, CRATE, TRACE, GRACE
- SPACE, PLACE, PEACE, REACT, TEACH
- REACH, BEACH, BREAK, GREAT, STEAM
- DREAM, CREAM, WORLD, HOUSE, LIGHT
- And many more...

The word list is designed to include common English words that are appropriate for the Wordle game format.
