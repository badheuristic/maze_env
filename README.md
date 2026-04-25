# maze_env

A grid-world reinforcement learning environment for testing and comparing RL control algorithms on maze navigation tasks.

## Overview

A 10×10 Tkinter-based maze where an agent learns to navigate from a start position to a goal while avoiding walls and pits. Intended as a framework for experimenting with and comparing RL algorithms.

**Reward structure:**
- Goal reached: +50
- Wall collision: −3 (agent reverts)
- Pit: −10 (terminal)
- Each step: −1

## Included Algorithms

| File | Algorithm |
|------|-----------|
| `RL_brainsample_qlearning.py` | Q-Learning (off-policy TD) |
| `RL_brainsample_sarsa.py` | SARSA (on-policy TD) |
| `RL_brainsample_expsarsa.py` | Expected SARSA |
| `RL_brainsample_EligTrace.py` | TD(λ) with eligibility traces |
| `RL_brainsample_MC.py` | Monte Carlo (template) |
| `RL_brainsample_wrong.py` | Intentionally flawed (for teaching) |
| `RL_brain.py` | Base class for custom implementations |

## Tasks

| Task | Start | Goal | Walls | Pits |
|------|-------|------|-------|------|
| 1 | [2,7] | [7,8] | 9 | 3 |
| 2 | [3,1] | [3,8] | 8 | 10 |
| 3 | [0,0] | [5,5] | 13 | 6 |

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
