
import numpy as np
import pandas as pd

class rlalgorithm:
    def __init__(self, actions, *args, **kwargs):
        """Your code goes here"""
        pass

    def choose_action(self, observation, **kwargs):
        """Your code goes here"""
        pass


    def learn(self, s, a, r, s_, **kwargs):
        '''Implementation the Monte Carlo algorithm. 
        Collect information about state-action pairs and their returns during the episode.
        Wait until the end of the episode, update the Q(S,A) state-action value table using the returns.
        In each transition, simply return the next state and an empty action.
        '''
        pass
