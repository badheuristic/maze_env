
import numpy as np
import pandas as pd


class rlalgorithm:
    def __init__(self, actions, gamma=.9, epsilon=.1, alpha=0.01, *args, **kwargs):
        """
        q-learning (off-policy TD control)
        value function Q(s,a) is a lookup table where
            - the rows are states (s) as string
            - the columns are actions (a) up/down/left/right
            - the values are the estimated Q-values
            - table starts out empty and we dynamically add states as we encounter them
        epsilon-greedy policy (self.epsilon)
        """
        self.display_name = "QLearning"
        self.maxq = 0
        self.delta = 0
        self.q_table = {}
        self.actions = actions
        self.num_actions = len(actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

    def choose_action(self, observation, **kwargs):
        """
        choose an action using epsilon-greedy policy
        if random number < epsilon, choose random action (explore)
        else extract q-values for current observation row and return action 
        with max q-value (exploit)
        if multiple actions have same max q-value, choose one randomly
        """
        self.check_state_exist(observation)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            state_actions = self.q_table[observation]
            max_q = max(state_actions.values())
            best_actions = [a for a in self.actions if state_actions[a] == max_q]
            return np.random.choice(best_actions)

    def learn(self, s, a, r, s_, **kwargs):
        """
        update-rule for q-learning
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        update q-value for current s,a by looking at next state s_ and grabbin
        the max q-value for that state, ignoring epsilon-greedy policy
        if the next state is a terminal state, the expected future value is 0
        """
        self.check_state_exist(s)
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * max(self.q_table[s_].values())
        else:
            q_target = r
        
        self.delta = abs(q_target - q_predict)
        self.maxq = max(self.maxq, q_target)
        
        self.q_table[s][a] += self.alpha * (q_target - q_predict)
        return s_, None
    
    def count_state(self, state):
        return len(self.q_table), state, 1
