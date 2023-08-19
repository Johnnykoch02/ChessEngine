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
                        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1),
                        nn.LeakyReLU(),
                        nn.Flatten(),
                        nn.Linear(in_features=64 * 64, out_features=512),
                        nn.LeakyReLU()
                        )
                    total_concat_size += 512
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
            encoded_tensor_list.append(extractor(observations[key]))
    
        return th.cat(encoded_tensor_list, dim=1)
    
    def get_feature_dim(self):
        return self._features_dim
    
class GrandMasterNetwork(nn.Module):
    def __init__(self, lr, checkpoint_dir= "Grand_Master_v_0_0_0", version='0.0.0', name='GrandMasterNetwork', ):
        super(GrandMasterNetwork, self).__init__()
        from src.Utils.imports import observation_space, action_space
        self.lr = lr
        self.observation_space = observation_space
        self.features_extractor =GrandMasterFeaturesExtractor(observation_space=self.observation_space)
        self.action_space = action_space
        
        self.loss = nn.MSELoss()
        self.optimizer = th.optim.RMSprop(self.parameters(), lr=lr)
    
        self.encoding_layers = nn.Sequential(
            nn.Linear(self.features_extractor.get_feature_dim(), 1864),
            nn.LeakyReLU(),
            nn.Linear(1864, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 216),
            nn.LeakyReLU(),
        )

        self.move_layers = nn.Sequential(
            nn.Linear(216, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            # nn.Softmax()
        )
        
    def forward(self, observations):
        # Pass the observations through the encoding layers
        features = self.features_extractor(observations)
        encoded_layer = self.encoding_layers(features)
        # Get the piece layer from the next layers
        # pieces = self.piece_layers(encoded_layer)
        # Concat the piece layer to the encoded layer and pass that through the move layers
        # moves = self.move_layers(th.cat((encoded_layer, pieces), dim=1))
        moves = self.move_layers(encoded_layer)
        return moves
        
        
        
        
class GrandMasterValueAproximator(nn.Module):
    def __init__(self, observation_space, device='cuda'):
        super(GrandMasterValueAproximator, self).__init__()         
        self.observation_space = observation_space
        self.checkpoint_file = "./GM_ValueAproximator_Weights.zip"
        self.device = device
        self.to(self.device)
        self.residuals = [
            nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding='same'),
                        nn.Tanh(),
                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,4), padding='same'),
                        nn.Tanh(),
                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding='same'),
                        nn.Tanh(),
                        nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3, 3), padding='same'),
                        nn.Tanh(),
                        nn.BatchNorm2d(3),
                      )
            for _ in range(3)]
        for r in self.residuals:
            r.to(self.device)
            
        self.compressor = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=0),
                        nn.Tanh(),
                        nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), padding=0),
                        nn.Tanh(),
                        nn.Flatten()
                      ) # Output shape: torch.Size([10, 128*4*4])
        self.compressor.to(self.device)
        
        self.output_block = nn.Sequential(
            nn.Linear(128*4*4 + 4, 1028),
            nn.LeakyReLU(),
            nn.Linear(1028, 1028),
            nn.LeakyReLU(),
            nn.Linear(1028, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.output_block.to(self.device)

    def forward(self, obs):
        x = obs['board_state']
        for residual in self.residuals:
            x = x+residual(x)
        x = self.compressor(x)
        x = th.cat([x, obs['team_color'], obs['score'], obs['check']], dim=1)
        x = self.output_block(x)
        return x
    
    @staticmethod
    def Load_Model(model_path):
        model = GrandMasterValueAproximator(None)
        model.load_state_dict(th.load(model_path))
        model.eval()
        return model
    
    def save_checkpoint(self, file=None):
        print('[GrandMasterValueAproximator] Saving Checkpoint...')
        if file != None:
            th.save(self.state_dict(), file)
        elif self.checkpoint_file != None:
            th.save(self.state_dict(), self.checkpoint_file) 
    
    