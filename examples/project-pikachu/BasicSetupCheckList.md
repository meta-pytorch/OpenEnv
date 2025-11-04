#  Basic Pokémon RL Training Environment Setup

## 🖥️ System Requirements

- **Python**: 3.9 or higher (✅ 3.10 recommended)  
- **Node.js**: v20 or higher (for Pokémon Showdown server)  
- **npm**: Latest version  
- **Git**  
- **Operating System**: Windows 10/11, Linux, or macOS  

> ⚠️ Ensure Python and Node.js are added to your system's `PATH`.  
> ✅ Enable `pip` and Python launcher (`py`) during installation.

---

## 🧰 Hardware Recommendations

| Component | Requirement |
|----------|-------------|
| RAM      | 8–16 GB minimum |
| Storage  | At least 5 GB free |
| CPU      | Multi-core (4+ cores recommended) |
| GPU      | Recommended for faster training |

---

## ⚙️ Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3.10 -m venv venv
source venv/bin/activate
```

### 2. Upgrade Core Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install Required Packages

#### Option A: Manual Installation

```bash
# Install poke-env first to set compatible gymnasium version
pip install poke-env

# Core RL and utility libraries
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3==2.2.1  # or 2.3.2
pip install gymnasium
pip install poke-env  # Reinstall to ensure compatibility
pip install tensorboard matplotlib pandas wandb
pip install trl==0.19.1
pip install torchvision torchaudio
```

#### Option B: Install Specific poke-env Version

```bash
pip install poke-env==0.9.0
```

This will automatically install:

- `gymnasium>=1.0.0` – RL environment interface  
- `numpy>=2.0.2` – Numerical computing  
- `orjson>=1.24.3` – Fast JSON parsing  
- `pettingzoo>=2.32.3` – Multi-agent RL environments  
- `tabulate==0.9.0` – Table formatting  
- `websockets==15.0.1` – WebSocket client  

---

## ✅ Test Your Setup

Create a basic test script to verify installation.  
Example script:

![Test Script Output](https://github.com/user-attachments/assets/c5cb5911-388b-4511-8d43-0a48f5e047a0)

---

## 🐛 Known Issues

- Compatibility issues between versions of:
  - `stable-baselines3` and `poke-env`
  - `gymnasium`, `pettingzoo`, and `stable-baselines3`

---

## 🛠️ Fixes Applied

- Ensured Python 3.10 is selected for `venv`
- Uninstalled conflicting packages:

```bash
pip uninstall -y poke-env gymnasium pettingzoo stable-baselines3 trl torch
```

- Installed newer compatible version of `stable-baselines3`:

```bash
pip install stable-baselines3>=2.4.0
```

---

Let me know if you'd like this turned into a shareable README or setup script!
