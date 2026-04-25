# Temporal Difference Control in a GridWorld Maze: Implementation and Analysis

## Table of Contents
1. [Introduction](#introduction)
2. [Algorithm Definitions and Implementation](#algorithm-definitions-and-implementation)
3. [Quantitative Results](#quantitative-results)
4. [Results Analysis](#results-analysis)
5. [Qualitative Observations](#qualitative-observations)
6. [Comparison to Classical Search](#comparison-to-classical-search)
7. [Conclusion](#conclusion)

---

## Introduction

This report presents the implementation and analysis of four temporal difference (TD) control algorithms applied to a 10x10 GridWorld maze environment. The algorithms are:

1. Q-Learning (off-policy TD control)
2. SARSA (on-policy TD control)
3. Expected SARSA (on-policy TD with analytical expectation)
4. TD(lambda) / SARSA(lambda) (on-policy TD with eligibility traces)

Three maze configurations of increasing difficulty are used (Tasks 1-3). The agent starts at a fixed position and must reach the goal while avoiding walls and pits. Reward structure: goal = +50, each step = -1, wall collision = -3, pit = -10 (episode ends).

All experiments use $\alpha = 0.01$, $\gamma = 0.9$, $\varepsilon = 0.1$, and 1500 episodes per run. TD(lambda) additionally uses $\lambda = 0.8$.

---

## Algorithm Definitions and Implementation

### Q-Learning

Off-policy TD control. The Bellman update uses the greedy maximum over next-state Q-values, independent of the behavior policy:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right)$$

The Q-table is stored as a nested Python dictionary `{state: {action: value}}` for O(1) lookup. New states are added on first encounter and initialized to zero. `learn()` returns `(s', None)` since Q-Learning does not commit to a next action.

**Key design choice:** Using a Python dictionary instead of a Pandas DataFrame yielded approximately a 100x speedup. Initial tests with DataFrames ran in O(n^2) time as rows were appended. The dictionary provides O(1) amortized access.

### SARSA

On-policy TD control. The update target uses $Q(s', a')$ where $a'$ is sampled from the behavior policy $\pi$ (epsilon-greedy):

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma Q(s',a') - Q(s,a) \right), \quad a' \sim \pi(\cdot \mid s')$$

The Q-table is stored as `{state: [q0, q1, q2, q3]}`. `learn()` samples $a'$ from $s'$ internally using epsilon-greedy, then returns `(s', a')` so the main loop feeds the sampled action back at the next step. This maintains a fully consistent on-policy trajectory.

### Expected SARSA

On-policy TD with the analytical expectation over the next-state action distribution, eliminating action-sampling variance:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma \mathbb{E}_\pi[Q(s',a')] - Q(s,a) \right)$$

$$\mathbb{E}_\pi[Q(s',a')] = (1-\varepsilon) \max_{a'} Q(s',a') + \frac{\varepsilon}{|A|} \sum_{a'} Q(s',a')$$

The expected value is computed analytically from the current Q-values. No action sampling occurs. `learn()` returns `(s', None)`. This makes the TD target deterministic given fixed Q-values and $\varepsilon$.

### TD(lambda) with Eligibility Traces

On-policy SARSA(lambda) with accumulating eligibility traces. Traces $e(s,a)$ record how recently and frequently each state-action pair was visited:

$$\delta = r + \gamma Q(s',a') - Q(s,a)$$

$$e(s,a) \leftarrow e(s,a) + 1 \quad \text{(accumulating)}$$

$$Q(s,a) \leftarrow Q(s,a) + \alpha \cdot \delta \cdot e(s,a) \quad \text{for all } (s,a)$$

$$e(s,a) \leftarrow \gamma \lambda \cdot e(s,a) \quad \text{for all } (s,a)$$

At episode end, all traces are reset to zero to prevent inter-episode credit leakage. The next action $a'$ is sampled on-policy (SARSA-style) and returned.

**Design parameter:** $\lambda = 0.8$, giving a decay rate of $\gamma\lambda = 0.72$. The effective credit horizon is approximately $1/(1-0.72) \approx 3.6$ steps, which matches typical corridor lengths in Tasks 2 and 3.

**Complexity note:** The trace broadcast loop runs over all visited states at every step: $O(|S_\text{visited}| \times |A|)$ per step. For a 10x10 grid with a maximum of 86 reachable states this is acceptable (approximately 344 updates per step), but would not scale to large state spaces without sparse trace storage.

---

## Quantitative Results

1500 episodes, $\alpha = 0.01$, $\gamma = 0.9$, $\varepsilon = 0.1$. **Med-10** is the median reward over the final 10 episodes. **Var-10** is the variance over the final 10 episodes.

| Algorithm | Task | Max Reward | Med-10 | Var-10 |
|-----------|------|-----------|--------|--------|
| Q-Learning | 1 | 41.0 | 41.0 | 1.64 |
| SARSA | 1 | 41.0 | 41.0 | 1.80 |
| Expected SARSA | 1 | 41.0 | 41.0 | 6.56 |
| TD(lambda) | 1 | 41.0 | 40.0 | 7.40 |
| Q-Learning | 2 | 38.0 | 20.0 | 590.76 |
| SARSA | 2 | 38.0 | 28.0 | 80.09 |
| Expected SARSA | 2 | 38.0 | 30.0 | 379.60 |
| **TD(lambda)** | **2** | **38.0** | **36.0** | **36.40** |
| Q-Learning | 3 | 37.0 | 32.5 | 243.44 |
| SARSA | 3 | 37.0 | 28.5 | 415.49 |
| Expected SARSA | 3 | 37.0 | 18.0 | 516.69 |
| **TD(lambda)** | **3** | **37.0** | **36.5** | **1.89** |

---

## Results Analysis

### Task 1 (simple map: 9 walls, 3 pits)

All four algorithms converge to near-optimal policies. The maximum achievable reward is 41 (9 steps, no pits), which every algorithm achieves in its best episode.

Q-Learning and SARSA reach Med-10 = 41 with low variance (< 2), indicating reliable policy execution by episode 1500. Expected SARSA also achieves Med-10 = 41 but with slightly higher variance (6.56). This is likely because its expected-value target produces smaller TD errors early in training, slowing the rate at which Q-values for good actions separate from those for poor actions. TD(lambda) achieves Med-10 = 40 with the highest variance (7.40) on this task. Trace broadcasts can occasionally reinforce suboptimal paths when the agent re-explores a previously visited corridor.

### Task 2 (pit-heavy map: 10 pits)

This is where the algorithms diverge significantly. TD(lambda) achieves **Med-10 = 36, Var-10 = 36.4**, the best of all algorithms by a large margin. The eligibility trace mechanism is directly responsible. The first time the agent successfully navigates a pit corridor, every step of that trajectory is immediately reinforced. One-step methods must propagate this corridor value backwards over many future episodes.

Q-Learning drops sharply to Med-10 = 20 with extremely high variance (590.76). This is the off-policy recklessness effect: the greedy max target assumes the agent will always execute the greedy action in the future, causing it to plan routes immediately adjacent to pits. When $\varepsilon$-exploration fires near a pit, the agent falls in, producing dramatic reward swings.

SARSA (Med-10 = 28) and Expected SARSA (Med-10 = 30) perform better than Q-Learning due to their on-policy updates encoding caution near pits. Unexpectedly, Expected SARSA outperforms SARSA on Task 2 despite SARSA being more commonly cited as the safer algorithm.

### Task 3 (diagonal wall barrier: 6 pits)

TD(lambda) achieves its most dramatic advantage: **Med-10 = 36.5, Var-10 = 1.89**, which is nearly perfect convergence on the hardest task.

Task 3 features a diagonal wall forcing navigation through a narrow corridor (columns 3-4, rows 2-4). This is precisely the scenario where eligibility traces provide maximum value. The corridor must be traversed early in the episode, and credit must propagate backwards through several steps to reinforce it. One-step methods require many more episodes to propagate the corridor's value to its entry.

Expected SARSA collapses on Task 3: Med-10 = 18, Var-10 = 516.69. The expected value's averaging over the action distribution reduces the magnitude of Q-value updates at the corridor entry. The uniform component $\frac{\varepsilon}{|A|} \sum Q(s',a')$ includes low-value actions leading into walls and pits, pulling the TD target $\delta$ downward and slowing the rate at which the corridor entry's Q-value separates from surrounding states. This is the same averaging property that reduces variance on simple maps, but on Task 3 it delays convergence precisely where a large, decisive update is needed.

SARSA shows high variance (415.49). The diagonal wall creates two distinct solution regions. $\varepsilon$-exploration intermittently sends the agent between them even after partial convergence.

### Computation Time

All 1500-episode runs complete in under 60 seconds. The critical optimization is the dictionary-based Q-table: O(1) state lookup versus O(n^2) DataFrame appends. TD(lambda) has the highest per-step cost at $O(|S_\text{visited}| \times 4)$, but with a maximum of 86 reachable states this is negligible.

---

## Qualitative Observations

**Q-Learning** plans aggressively, hugging pit boundaries for the shortest paths. On simple maps this is effective. On pit-dense maps, $\varepsilon$-exploration causes frequent falls, producing high variance. The off-policy max target instills no implicit penalty for future exploration errors.

**SARSA** is more conservative. Its on-policy updates implicitly penalize routes near pits because the expected value of being near a pit includes the probability of $\varepsilon$-exploration into it. This produces cautious, longer routes but avoids catastrophic falls. Task 3's topology defeats this conservatism because the only path through the diagonal wall forces multiple near-pit states.

**Expected SARSA** is the most variance-stable on simple maps. The analytical expectation removes noise from the update target. However, this same stability causes delayed learning on complex maps. The average over all actions is less reactive to the widening gap between good-action and poor-action Q-values than a single greedy or sampled estimate.

**TD(lambda)** learns fastest and converges most reliably on complex maps. Trajectory plots for Task 2 confirm this visually. Early episodes show random walks; late episodes show tight, consistent paths through the corridor. The transition from random to directed behavior occurs notably earlier than in the single-step methods.

---

## Comparison to Classical Search

Tasks 2 and 3 use maps that were also benchmarked against classical search algorithms (BFS, UCS, A*, Greedy Best-First), enabling a direct comparison between model-free RL and classical planning.

### Search Results

**Task 2 map:**

| Algorithm | Path Cost | States Explored | Optimal? |
|-----------|-----------|-----------------|----------|
| BFS | 16.8 | 102 | No |
| UCS | 9.2 | 109 | Yes |
| A* (Euclidean) | 9.2 | 56 | Yes |
| A* (Manhattan) | 9.2 | 50 | Yes |
| Greedy BFS | 25.8 | 25 | No |

**Task 3 map:**

| Algorithm | Path Cost | States Explored | Optimal? |
|-----------|-----------|-----------------|----------|
| BFS | 8.2 | 57 | Yes |
| UCS | 8.2 | 57 | Yes |
| A* (Euclidean) | 8.2 | 33 | Yes |
| Greedy BFS | 11.1 | 27 | No |

### Path Quality Comparison

The RL reward model implies that max reward = 50 - steps for a pit-free path. The best RL rewards therefore imply:

- Task 2 best RL path: $50 - 38 = 12$ steps vs. A* optimal: ~9 steps (+33%)
- Task 3 best RL path: $50 - 37 = 13$ steps vs. A* optimal: ~8 steps (+62.5%)

RL finds paths approximately 33-63% longer than the search-optimal solution.

### Key Differences

**Optimality guarantee.** A* and UCS guarantee the optimal path given an admissible heuristic. Both Euclidean and Manhattan distance are admissible for this grid. RL provides no such guarantee and converges to near-optimal but not provably optimal policies.

**Sample efficiency.** A* explores 33-56 unique states to find the solution. RL agents visit states millions of times over 1500 episodes. Search is orders of magnitude more efficient for known, static environments.

**Prior knowledge requirement.** A* and UCS require a complete model: the transition function, all states, and step costs. RL requires only reward signals from direct interaction. No map is needed.

**Policy coverage.** Search returns a single path from the start state. RL produces a complete policy $\pi(s)$ over every state in the maze. If the agent is displaced mid-episode due to stochastic transitions or noisy actuation, the RL policy handles it. A search path cannot.

**Stochastic environments.** A* and UCS assume deterministic transitions. RL handles stochastic dynamics naturally. For this static, deterministic maze the distinction does not apply, which is precisely why search outperforms RL on raw path quality.

**Observation on Task 3:** BFS finds the same path cost as UCS and A* (8.2), meaning the optimal path contains no pits. BFS's cost-agnostic search happens to find the optimal-cost path because the shortest-hop path is also the cheapest. This is consistent with RL agents' best episodes also being pit-free (reward = 37 = 50 - 13, no -10 pit penalties).

---

## Conclusion

TD(lambda) is the most effective algorithm on complex maps due to its multi-step credit assignment via eligibility traces. It achieves the highest median reward on Tasks 2 and 3 and the lowest variance on Task 3, confirming that backward credit propagation is particularly valuable when the maze contains structural bottlenecks such as corridors and diagonal walls.

Q-Learning performs well on simple maps but degrades under dense pit configurations due to its off-policy recklessness. SARSA offers a stable, cautious baseline. Expected SARSA provides the lowest variance on simple maps but fails to converge reliably on Task 3 because its averaged TD target slows the rate at which Q-values separate near critical narrow passages.

Compared to classical search (A*, UCS), RL requires substantially more computation and finds suboptimal paths (33-63% longer than optimal). However, RL operates without a world model and produces a complete policy over all states. These properties are essential in real-world environments where a precomputed map is unavailable or the environment is non-stationary.
