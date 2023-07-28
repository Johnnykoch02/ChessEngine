from stable_baselines3 import PPO

from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy

import os
import sys
import time
import random

import gym

from .GrandMasterEnviornment import GrandMasterEnv
from .GrandMasterNetwork import GrandMasterNetwork
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.logger import configure


# set up logger
# Set new logger

import numpy as np
import torch as th
from collections import deque, namedtuple


class RolloutBuffer(object):
    def __init__(self, size=10000):
        self.size = size
        self.mem_cntr = 0
        self.transition_memory = []
        self.temp_transition = None

    def start_transition(self, state, action, reward, terminal):
        if len(state.shape[0] == 1):
            state = np.squeeze(state, axis=0)
        if len(action.shape > 1):
            action = np.squeeze(action, axis=0)
        
        self .temp_transition = {
            'state': state, 'action': action, 'reward': reward, 'terminal': terminal
        }

    def finish_transition(self, state_):
        if len(state_.shape[0] == 1):
            state_ = np.squeeze(state_, axis=0)
        self.temp_transition['state_'] = state_
        self.mem_cntr +=1 
        
    
