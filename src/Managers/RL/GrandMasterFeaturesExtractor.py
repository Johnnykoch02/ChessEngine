# from .custom_gym import BarrettHandGym
from stable_baselines3 import PPO
import os
from typing import Dict, List
import gym
from gym import spaces

from torch import nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    
class GrandMasterFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(GrandMasterFeaturesExtractor, self).__init__(observation_space, features_dim= 1)
        extractors = {}
        total_concat_size = 0               
        for key, subspace in observation_space.spaces.items():
                if key == 'board_state':
                    extractors[key] = nn.Sequential(
                        nn.Conv2d(2, 64, (3,3)),
                        nn.LeakyReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(32, 48,(3,3)),
                        nn.LeakyReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Flatten(),
                        nn.Linear(768, 128, activation='relu')
                        )
                    total_concat_size += 128
                elif key == 'team_color': 
                    extractors[key] = nn.Sequential(
                        nn.Linear(1, 4),
                        nn.Sigmoid(),
                        nn.Linear(4, 16),
                        nn.Sigmoid(),
                    )
                    total_concat_size += 16
                elif key == 'score':
                    extractors[key] = nn.Sequential(
                        nn.Linear(1, 4),
                        nn.Sigmoid(),
                        nn.Linear(4, 16),
                        nn.Sigmoid(),
                    )
                    total_concat_size += 16
                elif key == 'check':
                    extractors[key] == nn.Sequential(
                        nn.Linear(1, 8),
                        nn.Sigmoid(),
                        nn.Linear(8, 32),
                        nn.Sigmoid(),
                    )
                    total_concat_size += 32
   
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
    
    def forward(self, observations):
        encoded_tensor_list = []
        '''extractors contain nn.Modules that do all of our processing '''
        for key, extractor in self.extractors.items():
            # print('Key:', key, 'Extractor:', extractor)
            encoded_tensor_list.append(extractor(observations[key]))
    
        return th.cat(encoded_tensor_list, dim= 1)