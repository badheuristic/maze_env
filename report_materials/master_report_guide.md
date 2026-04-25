# master report guide — ece 657a assignment 2

this is a copy-paste blueprint for the final report. all data is filled in from the
experimental runs. sections are ordered to match the assignment rubric.

---

## 1. introduction

this assignment implements and evaluates four temporal difference (td) control algorithms
on a 10x10 gridworld maze: q-learning, sarsa, expected sarsa, and td(λ) with eligibility
traces. three increasingly complex maze configurations are used (tasks 1–3), each featuring
walls, pits, and a single goal state. the objective is to learn a policy that maximizes
expected cumulative reward — reaching the goal as quickly as possible while avoiding pits.

all algorithms share the following parameters: **α = 0.01, γ = 0.9, ε = 0.1** over
**1500 episodes** per run. td(λ) additionally uses **λ = 0.8**.

---

## 2. algorithm definitions and implementation

### 2.1 q-learning (`RL_brainsample_qlearning.py`)

off-policy td control. update rule:
```
Q(s,a) ← Q(s,a) + α(r + γ·max_a' Q(s',a') − Q(s,a))
```
uses the greedy maximum over next-state q-values as the td target, regardless of the
behavior policy. q-table stored as a nested python dict for O(1) state lookup.

### 2.2 sarsa (`RL_brainsample_sarsa.py`)

on-policy td control. update rule:
```
Q(s,a) ← Q(s,a) + α(r + γ·Q(s',a') − Q(s,a))
```
where a' ~ π(s') (epsilon-greedy). learn() samples a' from s', returns (s_, a_) so the
main loop can feed the sampled action back at the next step.

### 2.3 expected sarsa (`RL_brainsample_expsarsa.py`)

on-policy td with analytical expectation over the next-state action distribution:
```
Q(s,a) ← Q(s,a) + α(r + γ·E_π[Q(s',a')] − Q(s,a))
E_π[Q(s',a')] = (1−ε)·max_a' Q(s',a') + (ε/|A|)·Σ Q(s',a')
```
no action sampling in learn(). returns (s_, None). lower variance than sarsa by design.

### 2.4 td(λ) / sarsa(λ) (`RL_brainsample_EligTrace.py`)

on-policy td with accumulating eligibility traces. at each step:
```
δ = r + γ·Q(s',a') − Q(s,a)
e(s,a) ← e(s,a) + 1
Q(s,a) ← Q(s,a) + α·δ·e(s,a)   (for ALL s,a)
e(s,a) ← e(s,a)·γλ             (for ALL s,a)
```
traces reset to zero at episode end. λ=0.8 used (propagates credit ~5–8 steps back).

---

## 3. quantitative results

### 3.1 results table (1500 episodes, α=0.01, γ=0.9, ε=0.1)

| algorithm | task | max-rew | med-10 | var-10 | max-ep-len |
|-----------|------|---------|--------|--------|------------|
| q-learning | 1 | 41.0 | 41.0 | 2.09 | 398 |
| sarsa | 1 | 41.0 | 39.0 | 1.64 | 463 |
| expected sarsa | 1 | 41.0 | 39.0 | **1.05** | 534 |
| td(λ) | 1 | 39.0 | 37.0 | 4.80 | **197** |
| q-learning | 2 | 38.0 | 33.0 | 413 | 471 |
| sarsa | 2 | 38.0 | 27.0 | 383 | 468 |
| expected sarsa | 2 | 38.0 | 22.0 | 411 | 399 |
| td(λ) | 2 | 38.0 | **36.0** | 559 | **282** |
| q-learning | 3 | 37.0 | 28.5 | 853 | 449 |
| sarsa | 3 | 37.0 | 24.5 | 1624 | 542 |
| expected sarsa | 3 | 37.0 | 1.5 | 484 | 583 |
| td(λ) | 3 | 37.0 | **36.0** | 527 | **258** |

### 3.2 plots to include

include the following plots from `data/` and `report_materials/`:
- `data/exp_T1_ep1k_*_jointplot.png` — reward + path length curves for task 1
- `data/exp_T1_ep1k_*_QLearning_traj.png` — q-learning trajectory heatmap task 1
- `data/exp_T1_ep1k_*_SARSA_traj.png` — sarsa trajectory heatmap task 1
- `report_materials/qlearning_task[1-3]_rewards.png` — q-learning per-task reward curves
- `report_materials/sarsa_task[1-3]_rewards.png` — sarsa per-task reward curves

all plots should include: axis labels (episode, reward/length), title, legend with
algorithm names.

---

## 4. results analysis

### 4.1 performance on task 1 (simple map)

task 1 is the simplest configuration: 8 wall cells, 3 pits, a relatively open layout.
all four algorithms converge to near-optimal policies by episode 1500.

- q-learning achieves the highest median (41.0) — its off-policy greedy target allows
  it to plan aggressively on a map where pits are sparse and far from the main path.
- expected sarsa achieves the lowest variance (1.05) due to its analytical averaging
  of the next-state action distribution.
- td(λ) converges fastest (max episode length 197 vs 398–534 for others) because
  eligibility traces propagate the first successful path's reward backwards immediately.

### 4.2 performance on task 2 (pit-heavy map)

task 2 has 10 pit cells arranged throughout the maze, creating several near-miss corridors.

- td(λ) dominates with median 36.0 — eligibility traces allow a single successful
  corridor traversal to immediately reinforce the entire path.
- q-learning drops to median 33.0, still performing well due to its aggressive
  off-policy updates.
- sarsa (27.0) and expected sarsa (22.0) both underperform — their on-policy caution
  near pits makes them overly conservative on routes that require precise navigation.
- variance is high across all algorithms (383–559) due to epsilon-exploration near pits.

### 4.3 performance on task 3 (complex diagonal wall)

task 3 is the hardest: a diagonal wall forcing navigation through a narrow corridor,
plus 6 pits in the lower half.

- td(λ) again dominates with median 36.0 and shortest episode lengths (258).
  the corridor-reinforcement mechanism is most valuable here.
- sarsa's variance explodes to 1623.84 — the diagonal wall creates two distinct
  solution regions and epsilon exploration sends the agent between them unpredictably
  even after partial convergence.
- expected sarsa collapses to median 1.5: the analytical expected value target
  dampens the gradient signal near the narrow corridor entry, stalling convergence.
- q-learning median of 28.5 is consistent with its moderate performance on all maps.

### 4.4 computation time

all runs complete in under 60 seconds for 1500 episodes. key optimization: q-tables
stored as python dicts (O(1) lookup) rather than pandas DataFrames. initial tests with
DataFrames were approximately 100x slower due to quadratic scaling with state count.

a second optimization: `time.sleep(0.2)` in `maze_env.py`'s `reset()` function was
moved inside the `if renderNow:` conditional, eliminating 15+ minutes of dead-time
during headless training runs.

td(λ) has the highest per-step cost: O(|states| × 4) for the trace broadcast loop
(~86 × 4 = 344 updates per step). this is acceptable here but would not scale to
continuous or large discrete state spaces without sparse trace data structures.

---

## 5. qualitative observations

**q-learning — reckless but fast**
q-learning treats its greedy target as if it will always execute optimally in the future.
this makes it bold near pits: it will hug a pit boundary if it's the shortest path,
banking on the 90% epsilon-greedy probability to not fall in. on dense pit maps, this
creates frequent accidental pit falls during testing (contributing to high variance).

**sarsa — cautious but slow on complex maps**
sarsa's on-policy updates encode a fear of pits proportional to epsilon. it avoids paths
adjacent to pits even when those paths are near-optimal. on task 1 this is fine; on task 3
this caution combined with the wall topology produces extreme variance as the agent
oscillates between partial solutions.

**expected sarsa — stable on simple maps, fragile on complex ones**
the variance reduction of expected sarsa is clearly visible on task 1 (1.05 vs 1.64 for
sarsa). however, the same smoothing mechanism hurts it on tasks 2 and 3 where sharp value
gradients near walls and corridors are needed for fast convergence.

**td(λ) — best on hard tasks, fastest learning overall**
eligibility traces make td(λ) the most practically effective algorithm on the complex maps.
the mechanism is intuitive: the first time the agent stumbles through the correct corridor,
every state on that path is immediately reinforced. this is qualitatively different from
the single-step backup of the other three algorithms. the shorter episode lengths across
all tasks confirm that td(λ) learns usable policies earlier in training.

---

## 6. comparison to search results

*(see `report_materials/comparative_analysis.md` for full data)*

tasks 2 and 3 use identical maps to asg1's task324 and task325 respectively.

| map | a* path cost | best rl path (equiv. steps) | gap |
|-----|-------------|----------------------------|-----|
| task324 / task 2 | 9.2 | 12 (max-rew = 38) | +30% |
| task325 / task 3 | 8.2 | 13 (max-rew = 37) | +59% |

**key differences:**

- **optimality**: a* guarantees the optimal path (admissible heuristic). all rl methods
  converge to near-optimal but not provably optimal policies. rl's best paths are
  30–60% longer than a*'s.
- **sample efficiency**: a* explores 33–56 unique states to find the solution. rl agents
  visit states millions of times across 1500 episodes before converging.
- **prior knowledge**: a* requires a complete model (transitions, costs). rl requires only
  rewards from direct interaction — no map needed.
- **policy coverage**: search returns one path from the start state. rl produces a complete
  policy over all states, enabling recovery from any position.
- **stochastic environments**: a* (and search in general) assumes deterministic transitions.
  rl handles stochastic environments naturally. for this static maze the distinction does
  not apply, which is why search outperforms rl on pure path quality.

---

## 7. conclusion

td(λ) is the strongest performer on complex maps due to its multi-step credit assignment.
q-learning performs well on simple and moderate tasks. expected sarsa offers the lowest
variance on simple tasks but is the weakest algorithm on complex ones. sarsa is a stable
baseline that underperforms td(λ) on all tasks but is less sensitive to topology than
expected sarsa.

compared to classical search, rl agents require substantially more computation and produce
suboptimal paths. however, rl operates without a world model and generalizes across all
states — properties that matter in real-world environments where a* is not applicable.
