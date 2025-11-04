# FastSVERL: Approximating Shapley Explanations in Reinforcement Learning

This repository provides the official implementation for the paper "Approximating Shapley Explanations in Reinforcement Learning" (NeurIPS 2025). It serves as both a framework for generating Shapley explanations and a guide to reproducing the paper's experiments.

## Project Overview

Reinforcement learning has achieved remarkable success in complex decision-making environments, yet its lack of transparency limits its deployment in practice, especially in safety-critical settings. Shapley values from cooperative game theory provide a principled framework for explaining reinforcement learning; however, the computational cost of Shapley explanations is an obstacle for their use. We introduce FastSVERL, a scalable method for explaining reinforcement learning by approximating Shapley values. FastSVERL is designed to handle the unique challenges of reinforcement learning, including temporal dependencies across multi-step trajectories, learning from off-policy data, and adapting to evolving agent behaviours in real time. FastSVERL introduces a practical, scalable approach for principled and rigorous interpretability in reinforcement learning.

## Installation

We recommend using Conda to replicate the environment precisely.

### Option 1: Conda (Recommended)

1.  Clone the repository:
    ```sh
    git clone https://github.com/djeb20/fastsverl.git
    cd fastsverl/fastsverl
    ```
2.  Create and activate the Conda environment from the `environment.yaml` file:
    ```sh
    conda env create -f environment.yaml
    conda activate fastsverl
    ```
3.  Install the `fastsverl` package in editable mode:
    ```sh
    pip install -e .
    ```

### Option 2: Pip and Venv

1.  Clone the repository:
    ```sh
    git clone https://github.com/djeb20/fastsverl.git
    cd fastsverl/fastsverl
    ```
2.  Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the dependencies from `requirements.txt`:
    ```sh
    pip install -r requirements.txt
    ```
4.  Install the `fastsverl` package in editable mode:
    ```sh
    pip install -e .
    ```

## Repository Structure

```text
fastsverl/                  # <-- Repository root
├── fastsverl/              # <-- Main project folder
│   ├── fastsverl/          # <-- The fastsverl Python package
│   │   ├── envs/           # Environments (Gridworld, Hypercube, etc.)
│   │   ├── characteristics.py  # Characteristic models
│   │   ├── dqn.py          # DQN agent implementation
│   │   ├── ppo.py          # PPO agent implementation
│   │   ├── models.py       # Core model architectures
│   │   ├── shapley.py      # Shapley explanation models
│   │   ├── training.py     # Reusable training routines
│   │   └── utils.py        # Utility functions
│   ├── environment.yaml    # Conda environment file
│   ├── requirements.txt    # Pip requirements file
│   └── setup.py            # Package setup file
├── experiments/            # All experiment scripts from the paper
│   ├── combined/           # Combined representations experiments
│   ├── fastsverl/          # Core FastSVERL experiments & qualitative results
│   ├── hypercube/          # Hypercube scaling experiments
│   ├── offpolicy/          # Off-policy learning experiments
│   ├── parallel/           # Parallel learning experiments
│   └── sampling/           # Sampling-based characteristics experiments
└── README.md               # This file
```

## Reproducing Experiments

The scripts in the `experiments/` directory are designed to reproduce all figures and results from the paper.

**General Workflow:**
Most experiments follow a multi-stage pipeline, typically:
1.  **Train an Agent:** An RL agent (e.g. DQN) is trained on an environment.
2.  **Train a Characteristic Model:** A characteristic model (e.g. behaviour) is trained using data from the agent.
3.  **Train a Shapley Model:** The final Shapley explanation model is trained using the agent and the characteristic model.

**Important Note:** The experiment scripts often launch multiple subprocesses by calling a run script (e.g., `run.py`, `run_policy_prediction.py`, `run_performance.py`). These subprocesses handle the individual training runs.

### Concrete Example: Behaviour Shapley Models in Mastermind

To replicate the experiment for training behaviour Shapley models for a DQN agent in Mastermind, follow these steps. These scripts train models across the different Mastermind environment sizes used in the paper.

1.  **Train the DQN agent:**
    This script trains the agent and saves the model to a timestamped directory (e.g. in `experiments/fastsverl/agents/runs/`).
    ```sh
    cd experiments/fastsverl/agents
    python dqn_mastermind.py
    ```

2.  **Train the behaviour characteristic:**
    Update `experiments/fastsverl/characteristics/dqn_policy_mastermind.py` with the agent's run name from step 1, then run it. This will train multiple characteristic models (e.g. 20).
    ```sh
    cd experiments/fastsverl/characteristics
    python dqn_policy_mastermind.py
    ```

3.  **Train the Shapley model:**
    Update `experiments/fastsverl/shapleys/dqn_policy_mastermind.py` with the agent's run name from step 1 and the run name of the *first* characteristic model from step 2, then run it.
    ```sh
    cd experiments/fastsverl/shapleys
    python dqn_policy_mastermind.py
    ```

All experiments include corresponding plotting scripts to reproduce the figures from the paper.

### Qualitative Results

For a detailed view of the qualitative results (e.g. behaviour, outcome, and prediction explanations for states in the Mastermind domains), please see the separate README located at:
`experiments/fastsverl/README.md`

## Using FastSVERL for Your Own Research

You can easily adapt FastSVERL to new environments and agents.

1.  **Custom Environments:**
    Implement the [Gymnasium API](https://gymnasium.farama.org/). See `fastsverl/envs/` for examples.
    ```python
    import gymnasium as gym
    class MyEnv(gym.Env):
        ... # implement required methods (reset, step, etc.)
    ```

2.  **Custom Agents:**
    Follow the agent structure (e.g. DQN, PPO) in `fastsverl/fastsverl/`. You can use the existing agents on your new environment or implement your own.

3.  **Custom Experiments:**
    Adapt the scripts in the `experiments/` directory to create your own training pipelines. For example, you can base your work on the agent -> characteristic -> Shapley model pipeline to generate explanations for your new components.

## Contributing and Support

We welcome issues and pull requests! Please follow the existing code style (docstrings, comments) and open an issue to discuss significant changes. For help or questions, please open an issue in the repository.

## Citation

If you use FastSVERL in your research, please cite our paper:

```bibtex
@inproceedings{beechey2025fastsverl,
  title={Approximating Shapley Explanations in Reinforcement Learning},
  author={Beechey, Daniel and {\c{S}}im{\c{s}}ek, {\"O}zg{\"u}r},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
