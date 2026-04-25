import numpy as np


class rlalgorithm:
    def __init__(self, actions, gamma=0.9, alpha=0.01, epsilon=0.1, *args, **kwargs):
        """
        expected sarsa (on-policy td control with expected value target)
        value function Q(s,a) stored as a nested dict: {state: [q_a0, q_a1, ...]}
        new states are added dynamically and initialized to zero.

        instead of sampling a' ~ pi(s') and using Q(s',a') as the target,
        we compute the full expectation E_pi[Q(s',a')] analytically.
        under epsilon-greedy this is:
            E_pi[Q(s',a')] = (1 - epsilon) * max_a' Q(s',a')
                           + epsilon / |A| * sum_a' Q(s',a')
        this removes one source of variance vs sarsa while staying on-policy,
        unlike q-learning which is off-policy (uses greedy max target).
        """
        self.display_name = "Expected SARSA"
        self.q_table = {}
        self.actions = actions
        self.num_actions = len(actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.maxq = 0
        self.delta = 0

    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_actions

    def choose_action(self, observation, **kwargs):
        """
        epsilon-greedy action selection.
        with probability epsilon choose a random action (explore),
        else choose the action with the highest q-value (exploit).
        ties in q-values are broken by random selection.
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
        expected sarsa update:
            Q(s,a) <- Q(s,a) + alpha * (r + gamma * E_pi[Q(s',a')] - Q(s,a))

        expected value under epsilon-greedy policy over next state s':
            E_pi[Q(s',a')] = (1 - epsilon) * max_a' Q(s',a')
                           + (epsilon / |A|) * sum_a' Q(s',a')

        no action needs to be sampled from s', so None is returned as a_.
        if s_ is terminal the future value term is dropped (q_target = r).
        """
        self.check_state_exist(s)
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]

        if s_ != 'terminal':
            q_next = self.q_table[s_]
            max_q_next = max(q_next)
            mean_q_next = sum(q_next) / self.num_actions
            # expected value under epsilon-greedy: greedy component + uniform component
            expected_q = (1.0 - self.epsilon) * max_q_next + self.epsilon * mean_q_next
            q_target = r + self.gamma * expected_q
        else:
            q_target = r

        self.delta = abs(q_target - q_predict)
        self.maxq = max(self.maxq, q_target)
        self.q_table[s][a] += self.alpha * (q_target - q_predict)
        return s_, None

    def count_state(self, state):
        return len(self.q_table), state, 1
