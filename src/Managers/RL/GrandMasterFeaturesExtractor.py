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
                        # nn.BatchNorm2d(subspace.shape),
                        nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3,3), padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3,3), padding=1),
                        nn.LeakyReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Flatten(),
                        )
                    total_concat_size += 128 * 4 * 4
                elif key == 'team_color': 
                    extractors[key] = nn.Sequential(
                        nn.Linear(1, 4),
                        nn.LeakyReLU(),
                        nn.Linear(4, 16),
                        nn.LeakyReLU(),
                    )
                    total_concat_size += 16
                    ''''''
                elif key == 'score':
                    extractors[key] = nn.Sequential(
                        nn.Linear(1, 4),
                        nn.LeakyReLU(),
                        nn.Linear(4, 16),
                        nn.LeakyReLU(),
                    )
                    ''''''
                    total_concat_size += 16
                elif key == 'check':
                    extractors[key] = nn.Sequential(
                        nn.Linear(2, 16),
                        nn.LeakyReLU(),
                    )
                    total_concat_size += 16
   
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
    
    def forward(self, observations):
        encoded_tensor_list = []
        '''extractors contain nn.Modules that do all of our processing '''
        for key, extractor in self.extractors.items():
            # print('Key:', key, 'Extractor:', extractor)
            # encoded_tensor_list.append(th.unsqueeze(extractor(observations[key]), dim=0))
            # tensor = th.unsqueeze(observations[key], dim=0)
            encoded_tensor_list.append(extractor(observations[key]))
    
        return th.cat(encoded_tensor_list, dim=1)