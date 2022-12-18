from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from .GrandMasterEnviornment import GrandMasterEnv
import numpy as np
import torch as th

import numpy as np
import torch as th
from torch.nn import functional as F


class AgentPtr:
    def __init__(self, model, env=None, team=None):
        from src.Utils.imports import Piece
        self.model = model
        self.env = env
        self.next = None
        self.team = team


class GrandMasterJudge:
    def __init__(self, agent1:AgentPtr, agent2:AgentPtr, num_games=10):
        self.agent1 = agent1
        self.agent2 = agent2
        self.current_agent = agent1
        self.agent1.next = agent2
        self.agent2.next = agent1
        
        from src.Utils.imports import Board, MoveGenerator, Piece
        self.Board = Board
        self.Piece = Piece
        self.MoveGenerator = MoveGenerator
        self.board = self.Board()
        self.score_line = 0
        self.num_games = num_games
        
        agent1.team = Piece.Color.WHITE
        agent2.team = Piece.Color.BLACK
        
        self.agent1.env = GrandMasterEnv(board=self.board, team_color=self.agent1.team)
        self.agent2.env = GrandMasterEnv(board=self.board, team_color=self.agent2.team)

        
        

class GrandMasterPPO(PPO):
    '''
        Goal: Implement custom RL Training algorithm to target the game of chess through the GrandMasterEnv.
        Implementation:
            1. https://github.com/DLR-RM/stable-baselines3/blob/002850f8ace0e045f7e9d370149a6fbb6cbcebad/stable_baselines3/common/on_policy_algorithm.py#L20
            2. https://github.com/DLR-RM/stable-baselines3/blob/002850f8ace0e045f7e9d370149a6fbb6cbcebad/stable_baselines3/common/base_class.py#L367
        Github PPO Details the Paper Trail of how to accomplish this.
        - Needs:
            - Model Controllers: two models, 1 for each color, Model Controller will need to implement two GrandMasterPPOs and
            train after N number of steps, similar to how their algorithms already function. 
            - These Models are going to share a common enviornment
            - 
    '''
    
    def __init__(self, env, policy, learning_rate=5e-4, gamma=0.99, lam=0.95, clip_range=0.2, clip_range_vf=None,
                 n_steps=128, nminibatches=4, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 adaptive_kl_penalty=2.0, t_total=-1, policy_kwargs=None, tensorboard_log=None,
                 create_eval_env=False, seed=None, reward_scale=1.0, **kwargs):
        super(GrandMasterPPO, self).__init__(env, policy, learning_rate, gamma, lam, clip_range, clip_range_vf,
                                        n_steps, nminibatches, ent_coef, vf_coef, max_grad_norm,
                                        adaptive_kl_penalty, t_total, policy_kwargs, tensorboard_log,
                                        create_eval_env, seed, reward_scale, **kwargs)
        