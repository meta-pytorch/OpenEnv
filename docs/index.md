<div class="hero">
  <h1 class="hero__title"><img width="35" height="35" alt="OpenEnv logo" src="https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609" /> OpenEnv: Agentic Execution Environments</h1>
  <p class="hero__subtitle">
    An e2e framework for creating, deploying and using isolated execution environments for agentic RL training, built using Gymnasium style simple APIs.
  </p>
  <div class="hero__actions">
    <a class="hero__button hero__button--primary" href="quickstart/">
      Quick Start
    </a>
    <a class="hero__button" href="environment-builder/">
      Build Your Own Environment
    </a>
    <a class="hero__button" href="environments/">
      Explore Environments
    </a>
  </div>
  <div>
    <a href="https://discord.gg/YsTYBh6PD9"><img src="https://camo.githubusercontent.com/aa8bf380611b9abd47f42596a15b4842eaf01af84c86cc97001d2d5d166ef8c0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446973636f72642d4f70656e456e762d3732383964613f7374796c653d666c6174266c6f676f3d646973636f7264266c6f676f436f6c6f723d7768697465"></a>
    <a href="https://pypi.org/project/openenv-core/"><img src="https://img.shields.io/pypi/v/openenv-core?color=blue"></a>
    <a href="https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  </div>
</div>

## What is OpenEnv?

OpenEnv provides a standard for interacting with agentic execution environments via simple Gymnasium style APIs - `step()`, `reset()`, `state()`. Users of agentic execution environments can interact with the environment during RL training loops using these simple APIs.

In addition to making it easier for researchers and RL framework writers, we also provide tools for environment creators making it easier for them to create richer environments and make them available over familiar protocols like HTTP and packaged using canonical technologies like docker. Environment creators can use the OpenEnv framework to create environments that are isolated, secure, and easy to deploy and use.

## How can I contribute?

We welcome contributions from the community! If you find a bug, have a feature request, or want to contribute a new environment, please open an issue or submit a pull request. The repository is hosted on GitHub at [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv).

!!! warning
    OpenEnv is currently in an experimental stage. You should expect bugs, incomplete features, and APIs that may change in future versions. The project welcomes bug fixes, but to make sure things are well coordinated you should discuss any significant change before starting the work. It's recommended that you signal your intention to contribute in the issue tracker, either by filing a new issue or by claiming an existing one.

