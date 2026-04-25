# expected sarsa experimental findings (multi-task analysis)

## algorithm definition

expected sarsa is an **on-policy** td method that replaces the single sampled next-action
value in sarsa with the analytically computed expectation over the entire action distribution
under the current policy:

```
Q(s, a) <- Q(s, a) + alpha * (r + gamma * E_pi[Q(s', a')] - Q(s, a))
```

where the expected value under epsilon-greedy is:

```
E_pi[Q(s', a')] = (1 - epsilon) * max_a' Q(s', a')
                + (epsilon / |A|) * sum_a' Q(s', a')
```

this is a strict variance reduction over sarsa: instead of one noisy sample `Q(s', a')`,
we use the full weighted average. it remains on-policy (unlike q-learning) because the
expectation is taken under the behavior policy, not the greedy policy.

## implementation notes

- q-table stored as `{state: [q_a0, q_a1, q_a2, q_a3]}` matching the sarsa implementation
- `learn()` computes the expected value analytically — no action is sampled from `s'`
- returns `(s_, None)` because no next action is needed; the main loop calls `choose_action`
  independently for the next step
- terminal handling: `q_target = r` (no future value term)
- the greedy component `(1 - epsilon) * max_q` and uniform component `epsilon * mean_q`
  are summed; for epsilon=0.1 and 4 actions, this weights the best action at 92.5% and
  each other action at 2.5%

## quantitative results (alpha=0.01, gamma=0.9, epsilon=0.1, episodes=1500)

| task | max-rew | med-10 | var-10 | max-ep-len |
|------|---------|--------|--------|------------|
| task 1 (pillars) | 41.000 | 39.000 | 1.050 | 534 |
| task 2 (pit maze) | 38.000 | 22.000 | 410.810 | 399 |
| task 3 (complex) | 37.000 | 1.500 | 484.160 | 583 |

## qualitative behavioral analysis

### task 1 — lowest variance of all algorithms

expected sarsa achieves the **lowest variance on task 1 (1.05)**, outperforming all other
algorithms on this metric. this directly demonstrates the mathematical guarantee of expected
sarsa: by averaging out the action-sampling noise, the td target is more stable, producing
smoother q-table updates. the policy converges to an extremely consistent 9–11 step route.

the max episode length of 534 is the highest among all algorithms on task 1, reflecting that
expected sarsa explores more cautiously in early episodes — its expected value updates are
slightly more pessimistic than sarsa's sampled updates because the uniform component pulls
the target down slightly when the greedy action is strongly dominant.

### task 2 — median underperforms relative to theoretical advantages

on task 2, expected sarsa's median (22.0) is notably lower than both sarsa (27.0) and
q-learning (33.0). this is a key empirical finding that contradicts the naive expectation
that "lower variance = better performance."

the likely explanation: the expected value target mixes the greedy estimate with the
uniform average, which on a pit-heavy map means the target is partially discounted by
the q-values of actions leading into pits. even as the q-values for pit-adjacent states
become large and negative, they continue to drag down the expected target for otherwise
good state-action pairs nearby. this makes the policy over-cautious on task 2, avoiding
areas it should exploit once it has learned them.

### task 3 — severe underperformance

task 3 is expected sarsa's worst result: **median of 1.5**, effectively near-zero convergence.
the variance of 484 with a median near zero indicates the policy has not learned a reliable
solution path — it occasionally stumbles to the goal but has no consistent route.

the most likely cause is the interaction between the diagonal wall barrier in task 3 and the
expected value target. task 3 requires the agent to navigate through a narrow corridor (column
3–4, rows 2–4) early in the episode. the uniform-average component of the expected value
estimate makes the approach to this narrow corridor look less attractive than it truly is —
the q-values of the two "wrong" actions (wall/pit-adjacent) are not yet sufficiently negative
to push the greedy action's weight high enough to make the corridor-entry reliably preferred.
sarsa and q-learning do not have this problem because they evaluate either a single sampled
action or the strict maximum, both of which respond more sharply to the developing q-table
gradients near the corridor.

## comparison across tasks

| metric | task 1 | task 2 | task 3 |
|--------|--------|--------|--------|
| variance ranking (1 = lowest) | **1st** | 3rd | 2nd |
| median ranking (1 = highest) | 2nd (tied) | 4th | 4th |

expected sarsa shows a consistent tension: it is the most stable (lowest variance) on simple
tasks, but this same stability slows it down on complex maps with narrow critical paths or
dense pit fields. the variance reduction comes at the cost of slower gradient propagation
near topological bottlenecks.

## implementation consideration

returning `None` as the next action (instead of a sampled action like sarsa) is consistent
with the algorithm's semantics. expected sarsa does not commit to a next action at update
time — it integrates over all of them. the main loop's `choose_action` call at the start
of each step then performs a fresh epsilon-greedy selection, which is correct behavior.
