# maze_env

A grid-world reinforcement learning environment for benchmarking TD control algorithms on maze navigation tasks.

## Overview

A 10×10 maze where an agent learns to navigate from a fixed start to a goal while avoiding walls and pits. This project implements and compares four temporal difference (TD) control algorithms across three maze configurations of increasing difficulty.

**Algorithms:** Q-Learning, SARSA, Expected SARSA, TD(λ) with eligibility traces

**Key finding:** TD(λ) achieves the best convergence on complex maps (Med-10 = 36-36.5, Var-10 < 2 on Tasks 2-3). Q-Learning degrades sharply on pit-dense maps due to off-policy recklessness. See [report.md](report.md) for full analysis.

**Reward structure:**
- Goal reached: +50
- Wall collision: -3 (agent reverts)
- Pit: -10 (terminal)
- Each step: -1

## Included Algorithms

| File | Algorithm |
|------|-----------|
| `RL_brainsample_qlearning.py` | Q-Learning (off-policy TD) |
| `RL_brainsample_sarsa.py` | SARSA (on-policy TD) |
| `RL_brainsample_expsarsa.py` | Expected SARSA |
| `RL_brainsample_EligTrace.py` | TD(λ) with eligibility traces |
| `RL_brainsample_MC.py` | Monte Carlo (template) |
| `RL_brain.py` | Base class for custom implementations |

## Tasks

| Task | Start | Goal | Walls | Pits | Description |
|------|-------|------|-------|------|-------------|
| 1 | [2,7] | [7,8] | 9 | 3 | Open layout, sparse hazards |
| 2 | [3,1] | [3,8] | 8 | 10 | Pit-dense, near-miss corridors |
| 3 | [0,0] | [5,5] | 13 | 6 | Diagonal wall forcing narrow passage |

## Results Summary

Performance after 1500 episodes (α=0.01, γ=0.9, ε=0.1). **Med-10** = median reward over final 10 episodes; **Var-10** = variance over final 10 episodes.

| Algorithm | Task 1 Med-10 | Task 2 Med-10 | Task 3 Med-10 | Task 3 Var-10 |
|-----------|---------------|---------------|---------------|---------------|
| Q-Learning | 41.0 | 20.0 | 32.5 | 243.4 |
| SARSA | 41.0 | 28.0 | 28.5 | 415.5 |
| Expected SARSA | 41.0 | 30.0 | 18.0 | 516.7 |
| **TD(λ)** | **40.0** | **36.0** | **36.5** | **1.9** |

All algorithms reach max reward 41 (Task 1), 38 (Task 2), 37 (Task 3). TD(λ) dominates on Tasks 2-3.

## Setup

**pip:**
```bash
pip install -r requirements.txt
```

**Conda:**
```bash
conda env create -f environment_release.yml
conda activate rlcourse
```

**Nix:**
```bash
nix develop
```

## Usage

```bash
# Run with defaults (Task 1, 1500 episodes)
python run_main.py

# Custom episode count with rendering
python run_main.py 500 True

# Select task
TASK_NUM=2 python run_main.py
TASK_NUM=3 python run_main.py 1000
```

Results (learning curves, trajectory plots) are saved to `data/`.

## Implementing a Custom Algorithm

Subclass `RL_brain.RL` and implement three methods:

```python
from RL_brain import RL

class MyAgent(RL):
    def __init__(self, actions):
        super().__init__(actions)

    def choose_action(self, state):
        ...

    def learn(self, state, action, reward, next_state, done):
        ...
```

Then pass your agent instance to `run_main.py`'s training loop.
