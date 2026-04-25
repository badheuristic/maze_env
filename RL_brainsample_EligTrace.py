import numpy as np


class rlalgorithm:
    def __init__(self, actions, gamma=0.9, alpha=0.01, epsilon=0.1, lam=0.8, *args, **kwargs):
        """
        sarsa(lambda) — on-policy td control with accumulating eligibility traces.
        Q(s,a) stored as {state: [q_a0, q_a1, ...]} (dict of lists).
        e(s,a) traces stored in a parallel dict with the same structure.

        eligibility traces allow credit assignment backwards in time:
        every (s,a) pair keeps a decaying trace of how recently/often it
        was visited. the td error delta from each step is broadcast to all
        previously visited pairs, scaled by their trace value.

        the lambda parameter controls the trace decay rate:
            lambda=0  -> equivalent to one-step sarsa (no backward credit)
            lambda=1  -> approaches monte carlo (full episode credit)
        typical values 0.7-0.9 offer a bias-variance tradeoff sweet spot.

        accumulating traces: e(s,a) += 1 each visit (can exceed 1 for loops).
        traces decay every step by gamma*lambda and reset to zero at episode end
        so credit does not bleed across episode boundaries.
        """
        self.display_name = "TD(Lambda)"
        self.q_table = {}
        self.e_table = {}
        self.actions = actions
        self.num_actions = len(actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.lam = lam
        self.maxq = 0
        self.delta = 0

    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_actions
            self.e_table[state] = [0.0] * self.num_actions

    def choose_action(self, observation, **kwargs):
        """
        epsilon-greedy action selection.
        with probability epsilon choose a random action (explore),
        else choose the action with the highest q-value (exploit).
        ties in q-values broken randomly.
        """
        self.check_state_exist(observation)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_vals = self.q_table[observation]
            max_q = max(q_vals)
            best_actions = [a for a in self.actions if q_vals[a] == max_q]
            return np.random.choice(best_actions)

    def learn(self, s, a, r, s_, **kwargs):
        """
        sarsa(lambda) update rule (one step):

        choose a' from s' using epsilon-greedy (on-policy)
        td error:
               delta = r + gamma * Q(s',a') - Q(s,a)
           (if s' is terminal: delta = r - Q(s,a))
        accumulate trace for the visited pair:
               e(s,a) += 1
        update every known (s,a) pair proportional to its trace:
               Q(s,a) += alpha * delta * e(s,a)
               e(s,a) *= gamma * lambda
        if terminal, reset all traces to zero for next episode.

        the next action a_ is returned so run_main can feed it back
        at the next step without re-sampling (consistent with on-policy).
        """
        self.check_state_exist(s)
        self.check_state_exist(s_)

        # choose next action on-policy (sarsa-style)
        a_ = None
        if s_ != 'terminal':
            a_ = self.choose_action(s_)
            td_error = r + self.gamma * self.q_table[s_][a_] - self.q_table[s][a]
        else:
            td_error = r - self.q_table[s][a]

        # accumulate eligibility trace for current (s,a)
        self.e_table[s][a] += 1

        # broadcast td error to all (s,a) pairs weighted by their traces
        decay = self.gamma * self.lam
        for state in self.q_table:
            for action in self.actions:
                self.q_table[state][action] += self.alpha * td_error * self.e_table[state][action]
                self.e_table[state][action] *= decay

        # reset traces at end of episode so they don't leak into the next
        if s_ == 'terminal':
            for state in self.e_table:
                self.e_table[state] = [0.0] * self.num_actions

        self.delta = abs(td_error)
        self.maxq = max(self.maxq, self.q_table[s][a])
        return s_, a_

    def count_state(self, state):
        return len(self.q_table), state, 1
