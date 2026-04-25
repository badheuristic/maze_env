
import numpy as np
import pandas as pd


class rlalgorithm:
    def __init__(self, actions, gamma=0.9, alpha=0.01, epsilon=0.1, *args, **kwargs):
        """
        sarsa (on-policy td control)
        value function Q(s,a) is represented as a nested dictionary
        as the agent encounters new states it adds them to the dictionary
        new rows initialized to 0
        """
        self.q_table = {}
        self.actions = actions
        self.num_actions = len(actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.display_name = "SARSA"
        self.maxq = 0
        self.delta = 0
        
    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_actions

    def choose_action(self, observation, **kwargs):
        """
        select action using epsilon-greedy policy
        if random number > epsilon exploit else explore
        """
        self.check_state_exist(observation)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            state_actions = self.q_table[observation]
            max_q = max(state_actions)
            best_actions = [a for a in self.actions if state_actions[a] == max_q]
            return np.random.choice(best_actions)

    def learn(self, s, a, r, s_, **kwargs):
        """
        update Q(s,a) using sarsa update rule
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
        """
        self.check_state_exist(s)
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        a_ = None
        if s_ != 'terminal':
            a_ = self.choose_action(s_)
            q_target = r + self.gamma * self.q_table[s_][a_]
        else:
            q_target = r
        self.delta = abs(q_target - q_predict)
        self.maxq = max(self.maxq, q_target)
        self.q_table[s][a] += self.alpha * (q_target - q_predict)
        return s_, a_

    def count_state(self, state):
        return len(self.q_table), state, 1