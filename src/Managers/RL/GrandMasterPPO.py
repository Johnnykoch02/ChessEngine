from stable_baselines3 import PPO

from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy

import os
import sys
import time

from .GrandMasterEnviornment import GrandMasterEnv
from .GrandMasterFeaturesExtractor import GrandMasterFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor

import numpy as np
import torch as th

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
    
    def __init__(self, env=GrandMasterEnv(), learning_rate=5e-4, gamma=0.99, lam=0.95, clip_range=0.2, clip_range_vf=None,
                 n_steps=128, nminibatches=4, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 adaptive_kl_penalty=2.0, t_total=-1, policy_kwargs=dict(features_extractor_class=GrandMasterFeaturesExtractor,net_arch=[512, 256, dict(pi=[128,64], vf=[128,64])]), tensorboard_log=None,
                 create_eval_env=False, seed=None, reward_scale=1.0, **kwargs):
        super(GrandMasterPPO, self).__init__(env=env, policy=str('MultiInputPolicy'),verbose = 1,policy_kwargs= policy_kwargs, learning_rate=0.01, tensorboard_log=tensorboard_log)


class AgentPtr:
    def __init__(self, model, env=None, team=None):
        from src.Utils.imports import Piece
        self.model = model
        self.env = env
        self.next = None
        self.team = team


class GrandMasterJudge:
    def __init__(self, agent1:AgentPtr, agent2:AgentPtr, save_dir, log_dir, num_games=10, num_epochs=50):
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
        self.num_epochs = num_epochs
        self.last_obs = None
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        agent1.team = Piece.Color.BLACK
        agent2.team = Piece.Color.WHITE
        
        self.agent1.env = GrandMasterEnv(board=self.board, team_color=self.agent1.team)
        self.agent2.env = GrandMasterEnv(board=self.board, team_color=self.agent2.team)
        
    def _perform_losing_agents_pass_through(self):
        rollout_buffer = self.current_agent.next.model.rollout_buffer

        if rollout_buffer.size() == rollout_buffer.buffer_size-1:
            self.current_agent.next.model.train(rollout_buffer.size()//3, progress_bar=True)
            rollout_buffer.reset()
        ''' Update Variables for the current Model's Obersvations and Actions '''
        _, wk_check, wk_ckm = self.MoveGenerator.GenerateLegalMoves(self._board.white_king, self._board)
        _, bk_check, bk_ckm = self.MoveGenerator.GenerateLegalMoves(self._board.black_king, self._board)
        self.board.update_board_state(wk_check, wk_ckm, bk_check, bk_ckm)
        
        current_state = self.board.get_state(self.current_agent.next.team)
        obs = obs_as_tensor(current_state, device='cuda')
        action, values, log_prob = self.current_agent.next.model.policy.forward(obs, deterministic=True)
        action = action.cpu().numpy()
        _, reward, done, _ = self.current_agent.next.env.step(None, None, None, current_state['check'] )
        rollout_buffer.add(current_state, action, np.ones(1), values, log_prob)
    
    def convert_to_action(self, action):
        pass

                        
    def train_agents(self):
        ''' Number of Game Sets '''
        for _ in range(self.num_epochs):
            ''' Number of Games in each Set '''
            for _ in range(self.num_games):
                ''' Initialize the Game '''
                self.agent1.env.reset()
                self.agent2.env.reset()
                while not any(self.board.get_winner(self.Piece.Color.BLACK)):
                    rollout_buffer = self.current_agent.model.rollout_buffer

                    if rollout_buffer.size() == rollout_buffer.buffer_size -1:
                        self.current_agent.model.train(rollout_buffer.size()//3, progress_bar=True)
                        rollout_buffer.reset()
                    ''' Update Variables for the current Model's Obersvations and Actions '''
                    _, wk_check, wk_ckm = self.MoveGenerator.GenerateLegalMoves(self.board.white_king, self.board)
                    _, bk_check, bk_ckm = self.MoveGenerator.GenerateLegalMoves(self.board.black_king, self.board)
                    self.board.update_board_state(wk_check, wk_ckm, bk_check, bk_ckm)
                    
                    while True:
                        current_state = self.board.get_state(self.current_agent.team)
                        obs = obs_as_tensor(current_state, device='cuda')
                        action, values, log_prob = self.current_agent.model.policy.forward(obs, deterministic=True)
                        action = action.cpu().numpy()
                        piece = self.board.get_square((action[0], action[1]))
                        move = (action[2], action[3])
                        moveset, _, _ = self.MoveGenerator.GenerateLegalMoves(piece, self.board)
                        _, reward, done, info = self.current_agent.env.step(piece, move, moveset, current_state['check'] )
                        
                        rollout_buffer.add(current_state, action, np.ones(1), values, log_prob)
                        if done:
                            self._perform_losing_agents_pass_through()
                        elif info['valid_move']:
                            break
                        
                    time.sleep(0.5)
                    
                    self.current_agent = self.current_agent.next
                status = self.board.get_winner()
                '''                  Agent 1     Agent 2 '''
                self.score_line += status[1] - status[2]
                tmp = self.agent1.color
                self.agent1.color = self.agent2.color
                self.agent2.color = tmp
                
                self.current_agent = self.agent1 if self.agent1.team == self.Piece.Color.BLACK else self.agent2
            files = os.listdir(self.save_dir)
            try:
                os.remove(files[0])
            except:
                pass
            
            if self.score_line == 0:
                continue
            elif self.score_line > 0:
                self.agent1.model.save(self.save_dir)
            elif self.score_line < 0:
                self.agent2.model.save(self.save_dir)
                
            files = os.listdir(self.save_dir)
            self.agent1.model = GrandMasterPPO.load(files[0])
            self.agent2.model = GrandMasterPPO.load(files[0])
            self.score_line = 0
            
            self.agent1.team = self.Piece.Color.BLACK
            self.agent2.team = self.Piece.Color.WHITE
            
            self.agent1.env = GrandMasterEnv(board=self.board, team_color=self.agent1.team)
            self.agent2.env = GrandMasterEnv(board=self.board, team_color=self.agent2.team)
        
        self.agent1.model.learn(self.agent1.model.rollout_buffer)
        self.agent2.model.learn(self.agent2.model.rollout_buffer)
                
                            
