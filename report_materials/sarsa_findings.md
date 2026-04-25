# sarsa experimental findings (multi-task analysis)

## algorithm definition

sarsa (state-action-reward-state-action) is an **on-policy** td(0) control method.
the bellman update target is computed using the action actually selected by the behavior policy in the next state:

```
Q(s, a) <- Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))
```

the critical distinction from q-learning is that `a'` is drawn from the same epsilon-greedy policy being learned — not from the greedy max. this means sarsa evaluates the policy it actually follows, including its exploratory mistakes.

## implementation notes

- q-table stored as `{state: [q_a0, q_a1, q_a2, q_a3]}` (dict of lists, indexed by action int)
- new states initialized to all zeros when first encountered
- `choose_action` uses epsilon-greedy; ties broken by random selection among best actions
- `learn()` calls `choose_action(s_)` internally to select `a'`, then returns `(s_, a_)` so the main loop feeds the sampled action back at the next step — keeping the trajectory fully on-policy
- terminal states handled by setting `q_target = r` (no future value term)

## quantitative results (alpha=0.01, gamma=0.9, epsilon=0.1, episodes=1500)

| task | max-rew | med-10 | var-10 | max-ep-len |
|------|---------|--------|--------|------------|
| task 1 (pillars) | 41.000 | 39.000 | 1.640 | 463 |
| task 2 (pit maze) | 38.000 | 27.000 | 382.650 | 468 |
| task 3 (complex) | 37.000 | 24.500 | 1623.840 | 542 |

## qualitative behavioral analysis

### task 1 — stable convergence

sarsa converges reliably on task 1. the median reward of 39 with variance 1.64 indicates
a tightly consistent policy that almost always reaches the goal in 9–11 steps. the
on-policy nature of sarsa means the agent "knows" it will sometimes take exploratory steps
and programs caution into its policy accordingly. this is visible in the max episode length
(463 steps) — early training required extensive random exploration before the q-table filled in.

### task 2 — moderate conservatism near pits

on the pit-heavy task 2 map, sarsa's median drops to 27 (vs q-learning's 33 on the same run).
this might seem to favor q-learning, but the variance tells a different story: sarsa achieves
**382 vs q-learning's 413**. sarsa's on-policy updates make it slightly more cautious near
pit boundaries — it factors in the probability of epsilon-exploration into its value estimates,
which can make it avoid aggressive shortcuts adjacent to pits. however, this caution can also
cause it to take longer routes, dragging down the median.

### task 3 — high variance, partial convergence

task 3 produces sarsa's most striking result: variance of **1623.84**, the highest of all
four algorithms. this is unexpected for an on-policy method. the likely cause is the maze
topology: task 3 has walls forming a diagonal barrier (rows 4–6) that cuts the space, forcing
the agent into one of two corridors. epsilon exploration intermittently sends the agent down
the wrong corridor even late in training, and the long detour + possible pit encounters produce
extreme reward swings. sarsa's conservative value estimates slow convergence in this case because
the agent is uncertain about values in the alternate corridor and hesitates to commit.

## comparison across tasks

| metric | task 1 | task 2 | task 3 |
|--------|--------|--------|--------|
| max achievable reward (no pits, optimal path) | ~41 | ~38 | ~37 |
| median as % of max | 95.1% | 71.1% | 66.2% |
| variance trend | low | moderate | extreme |

the declining median-to-max ratio shows sarsa increasingly struggling as map complexity grows.
the extreme task 3 variance suggests the policy has not fully converged at 1500 episodes for
this map — longer training (2000–3000 episodes) would likely reduce the variance significantly.
