### Basic Pokemon RL Training ENV checklist

## System_Requirements : 
   a) Python: 3.9 or higher stable versions(3.10 recommended) 
   b) Node.js (v20 or higher, for Pokemon Showdown server)
   c) npm: Latest version
   d) Git
   e) OS:(Win10/11, Linux, macOS)

## Hardware_Recommendations :
   a) RAM: Min 8-16GB
   b) Storage: Minimum 5GB free space
   c) CPU: Multi-core processor(4 recommended)
   d) GPU: recommended for faster training

 Note: Make sure to install compatible python and node versions and add it in PATH in system variables.
 Enable pip & py launcher 

 ### Steps to setup:
     a) Create a virtual-environment (venv) in your project directory.( python -m venv venv for windows, or python 3.10 -m venv venv for Linux/Mac users)
     b) Activate venv (venv\Scripts\activate for Windows Users and source venv/bin/activate for Mac or Linux users) 
     c) Upgrade pip and install core tools:
                    python -m pip install --upgrade pip setuptools wheel
     d) The following packages: 
                     i) Install poke-env first (this sets the correct gymnasium version)
                    ii) pip install poke-env
                   iii) pip install torch --index-url https://download.pytorch.org/whl/cpu
                    iv) pip install stable-baselines3 (2.2.1 or 2.3.2)
                     v) pip install gymnasium
                    vi) pip install poke-env
                   vii) pip install tensorboard matplotlib pandas wandb
                  viii) pip install trl==0.19.1
                    ix) pip install torchvision torchaudio
     OR
     
     d) Install poke-env 0.9.0:
          pip install poke-env==0.9.0
     e) This automatically installs: 
                i) gymnasium&gt;=1.0.0(RL Env Interface)
               ii) numpy&gt;=2.0.2(numerical computing)
              iii) orjson&gt;=1.24.3(fast JSON parsing)
               iv) pettingzoo&gt;=2.32.3(HTTP library)
                v) tabulate;= 0.9.0(table formatting)
               vi) websockets== 15.0.1( WebSocket client) 
     f) Create a test script to verify setup and installation:
      Basic Example:
            <img width="741" height="296" alt="image" src="https://github.com/user-attachments/assets/c5cb5911-388b-4511-8d43-0a48f5e047a0" />

     
### Problems faced while installing packages and libraries(so far):
     a) stable versions of packages such as stable-baselines3, poke-env run into issues, no version compatibility
     b) gymnasium, pettingzoo, stable-baselines run into issues. 

### Fixes(so far):
    a) Select Python interpreter used to create venv(here 3.10), upgrade pip.
    b) Uninstall the packages; pip uninstall -y poke-env gymnasium pettingzoo stable-baselines3 trl torch
    c)  Install a newer stable-baselines3 that supports gymnasium>=1.0.0
           i) pip install stable-baselines3>=2.4.0


     
