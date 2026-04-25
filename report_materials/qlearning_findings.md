# q-learning experimental findings (multi-task analysis)

## quantitative sweep (tasks 1, 2, and 3)
*parameters: alpha=0.01, gamma=0.9, epsilon=0.1, episodes=1500*

**performance metrics:**
- **task 1:** `max-rew=41.000` | `med-10=39.000` | `var-10=2.760`
- **task 2:** `max-rew=38.000` | `med-10=20.500` | `var-10=521.360`
- **task 3:** `max-rew=37.000` | `med-10=28.500` | `var-10=723.840`

## qualitative behavioral analysis (model recklessness)
the most distinctive phenomenon observed in these runs is q-learning's extreme variance and "recklessness" on complex maps (tasks 2 and 3). 

because q-learning uses off-policy updates (via the `max()` bellman target), the agent fundamentally believes it will never make a mistake in the future. it has zero fear of pits or walls, so its q-table instructs it to walk completely adjacent to danger zones if it is the mathematically shortest path. 
however, its actual behavior policy is governed by 10% epsilon exploration. consequently, on maps 2 and 3 which are dense with pits, the agent attempts to walk optimally narrow paths, randomly triggers an epsilon exploration step, and hurls itself into a pit (-10 reward / termination). 

this creates the staggering statistical variance you see above (`521.0` and `723.0`). the agent clearly knows the optimal path (achieving highly successful `max-rew` scores in the high 30s), but its median test performance is severely hindered by its sheer recklessness.

## non-negotiable methodology optimizations
**1. structural paradigm shift ($O(N^2)$ to $O(1)$)**
initial trial runs using the `pandas.DataFrame` representation scaled quadratically, bringing processing hardware to a crawl. replacing `self.q_table` with a native python nested dictionary allowed all three 1500-episode mega sweeps to collectively execute in **under 5 total seconds**, enabling proper robust algorithmic scaling.

**2. environment simulation delays**
inside `maze_env.py` under the `reset()` function, there was an unconditionally invoked `time.sleep(0.2)`. over 4500 total episodes, this mathematically guaranteed the script would lag by 15 full minutes of pure dead-time explicitly waiting. this sleep threshold has been indented inside the `if renderNow:` block, guaranteeing fast headless-compute. 

## report material generation
i have successfully copied your generated reward convergence charts to this directory. you can now directly reference `qlearning_task1_rewards.png`, `qlearning_task2_rewards.png`, and `qlearning_task3_rewards.png` inside your final pdf submission.
