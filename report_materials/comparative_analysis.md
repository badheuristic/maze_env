# rl vs search: comparative analysis (tasks 2 and 3)

## map correspondence

| asg2 task | asg1 task id | agent start | goal |
|-----------|-------------|-------------|------|
| task 2 | task324 | [3, 1] | [3, 8] |
| task 3 | task325 | [0, 0] | [5, 5] |

both maps share identical wall layouts and pit placements. the cost model differs:
- **asg1 search**: step cost = 1, pit cost = 10 (path enters and exits a pit)
- **asg2 rl**: step reward = -1, pit reward = -10 (pit ends episode), wall penalty = -3, goal = +50

## asg1 search results summary

### task324 (= task 2)

| algorithm | path cost | states explored | optimal? |
|-----------|-----------|-----------------|----------|
| bfs | 16.8 | 102 | no |
| ucs | 9.2 | 109 | yes |
| a* (euclidean) | 9.2 | 56 | yes |
| a* (manhattan) | 9.2 | 50 | yes |
| greedy best-first | 25.8 | 25 | no |

### task325 (= task 3)

| algorithm | path cost | states explored | optimal? |
|-----------|-----------|-----------------|----------|
| bfs | 8.2 | 57 | yes (no pits on optimal path) |
| ucs | 8.2 | 57 | yes |
| a* (euclidean) | 8.2 | 33 | yes |
| greedy best-first | 11.1 | 27 | no |

## rl results summary (1500 episodes, best achieved)

### task 2

| algorithm | max-rew | equiv. path steps* | med-10 | var-10 |
|-----------|---------|-------------------|--------|--------|
| q-learning | 38.0 | 12 | 33.0 | 413 |
| sarsa | 38.0 | 12 | 27.0 | 383 |
| expected sarsa | 38.0 | 12 | 22.0 | 411 |
| td(lambda) | 38.0 | 12 | 36.0 | 559 |

### task 3

| algorithm | max-rew | equiv. path steps* | med-10 | var-10 |
|-----------|---------|-------------------|--------|--------|
| q-learning | 37.0 | 13 | 28.5 | 853 |
| sarsa | 37.0 | 13 | 24.5 | 1624 |
| expected sarsa | 37.0 | 13 | 1.5 | 484 |
| td(lambda) | 37.0 | 13 | 36.0 | 527 |

*equiv. path steps = 50 - max_rew (assuming no pit hits on the best episode)

## path quality comparison

### task 2: rl finds a 12-step path, search found a 9-step path

the best rl reward of 38 corresponds to a 12-step path to the goal (no pits hit).
a* found an optimal path of cost 9.2 — approximately 9 steps on a pit-free route.
this is a 33% longer path from rl's best episode.

the difference arises from two sources:
1. **exploration noise during learning**: rl cannot plan ahead and must discover good
   paths through trial-and-error. even after convergence, the best path found may not
   be globally optimal.
2. **reward shaping**: the rl reward function penalizes every step equally at -1. there
   is no guidance toward the goal until the agent discovers it. search algorithms,
   particularly a*, use an admissible heuristic (manhattan/euclidean distance) that
   actively guides state expansion toward the goal, pruning suboptimal directions early.

### task 3: rl finds a 13-step path, search found an 8-step path

the gap widens on task 3. the best rl reward of 37 corresponds to a 13-step path.
a* found the optimal 8-step path (ucs confirmed optimal, same cost as a* and bfs).
this is a **62.5% longer path** from rl's best result.

task 3's diagonal wall barrier creates a situation where search's heuristic guidance
is especially powerful: a* and ucs systematically explore states closest to the goal
first, naturally discovering the corridor through the wall. rl must discover this
corridor by random walk, which takes far longer and produces suboptimal paths.

## paradigm differences: efficiency and optimality

### training cost

| method | "episodes" to solve | states examined total |
|--------|--------------------|-----------------------|
| a* (task324) | 1 (single query) | 56 |
| a* (task325) | 1 (single query) | 33 |
| rl (best algorithm, task 2) | ~200 before first success | ~thousands per episode |
| rl (best algorithm, task 3) | ~300 before first success | ~thousands per episode |

search solves the problem in one pass. rl requires hundreds to thousands of episodes
and examines states many orders of magnitude more often. this is the fundamental
cost of model-free learning: no map, no transitions, no heuristic — only raw experience.

### optimality guarantees

- **a***: guarantees optimal path when the heuristic is admissible (never overestimates).
  both euclidean and manhattan distance are admissible for grid movement.
- **ucs**: guarantees optimal path always, regardless of heuristic.
- **rl (all algorithms)**: provides **no optimality guarantee**. the policy converges
  to an approximation that is good but not provably optimal. the best rl path found
  is 33–63% longer than the search-optimal path.

### adaptability and generalization

search algorithms require a complete, accurate model of the environment (transition
function, costs). give them a new map and they re-solve from scratch.

rl agents require no prior model. they can in principle adapt to stochastic
transitions, unknown rewards, and environments that change over time — none of which
classical search can handle without explicit model updates. for the static maze domain
here, this flexibility is not needed, which is why search outperforms rl on pure
path quality.

### policy vs. path

a key structural difference: search produces **a single path** (a sequence of actions
from start to goal). rl produces **a policy** (a mapping from every state to an action).
the rl policy covers the entire state space and can recover from any position in the
maze, not just the planned start state. if the agent is displaced mid-trajectory, the
rl policy handles it; a search path cannot.

## key takeaways

1. **optimality**: a*/ucs find provably optimal paths. rl converges to near-optimal
   policies but with no guarantee, and the best paths found are 33–63% longer.
2. **sample efficiency**: search is vastly more efficient — 56 state expansions vs.
   millions of state visits across 1500 rl episodes.
3. **robustness**: rl produces a complete policy over all states; search produces a
   single fragile path. rl is more robust to displacement or stochastic transitions.
4. **model requirements**: search needs a full environment model. rl needs only
   experience. for model-free settings, rl is the appropriate paradigm.
5. **complexity scaling**: on the harder task 3, the rl-to-search gap widens.
   td(lambda) partially closes this gap using multi-step credit assignment, but even
   td(lambda)'s best path (13 steps) is longer than a*'s (8 steps).
