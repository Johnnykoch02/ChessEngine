from stable_baselines3 import PPO

from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy

import os
import sys
import time

import gym

from .GrandMasterEnviornment import GrandMasterEnv
from .GrandMasterFeaturesExtractor import GrandMasterFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.logger import configure


# set up logger
# Set new logger

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
                 n_steps=64, nminibatches=1, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 adaptive_kl_penalty=2.0, t_total=-1, policy_kwargs=dict(features_extractor_class=GrandMasterFeaturesExtractor,net_arch=[512, 256, dict(pi=[128,64], vf=[128,64])]), tensorboard_log=None,
                 create_eval_env=False, seed=None, reward_scale=1.0, **kwargs):
        super(GrandMasterPPO, self).__init__(env=env, policy=str('MultiInputPolicy'),verbose = 1,policy_kwargs= policy_kwargs, learning_rate=0.01, tensorboard_log='.')


class AgentPtr:
    def __init__(self, model, env=None, team=None):
        from src.Utils.imports import Piece
        self.model = model
        self.env = env
        self.next = None
        self.team = team


class GrandMasterJudge:
    def __init__(self, agent1:AgentPtr, agent2:AgentPtr, save_dir, log_dir, draw_to_screen, num_games=10, num_epochs=50):
        self.agent1 = agent1
        self.agent2 = agent2
        self.current_agent = agent1
        self.agent1.next = agent2
        self.agent2.next = agent1
        logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        self.agent1.model.set_logger(logger)
        self.agent2.model.set_logger(logger)
        self.agent1.model.rollout_buffer.buffer_size = 64
        self.agent2.model.rollout_buffer.buffer_size = 64
        
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
        self.draw_to_screen = draw_to_screen
        
        agent1.team = Piece.Color.BLACK
        agent2.team = Piece.Color.WHITE
        
        self.agent1.env = GrandMasterEnv(board=self.board, team_color=self.agent1.team)
        self.agent2.env = GrandMasterEnv(board=self.board, team_color=self.agent2.team)
        
    def _perform_losing_agents_pass_through(self):
        rollout_buffer = self.current_agent.next.model.rollout_buffer
        policy = self.current_agent.next.model.policy

        if rollout_buffer.full:
                with th.no_grad():
                # Compute value for the last timestep
                    values = policy.predict_values(obs_as_tensor(self.board.get_state(self.current_agent.next.team), self.device))
                    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=np.array([0]))
                self.current_agent.next.model.train()
                rollout_buffer.reset()

        policy.set_training_mode(False)
        # Sample new weights for the state dependent exploration
        if self.current_agent.next.model.use_sde:
            policy.reset_noise(1)
        if self.current_agent.next.model.use_sde and self.current_agent.next.model.sde_sample_freq > 0:
                # Sample a new noise matrix
                policy.reset_noise(1)
        obs = self.board.get_state(self.current_agent.next.team)
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(obs, self.current_agent.model.device)
            actions, values, log_probs = policy(obs_tensor)
            
        actions = actions.cpu().numpy()
        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.current_agent.next.model.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.current_agent.next.model.action_space.low, self.current_agent.next.model.action_space.high)
        
        _, rewards, dones, infos = self.current_agent.next.env.step(None, None, None, obs['check'])
        # Give access to local variables
        
        if isinstance(self.current_agent.next.model.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)
        
        rollout_buffer.add(obs, actions, rewards, np.ones(shape=(1,)), values, log_probs)
    
    def collect_agent_rollout_buffer(self):
        rollout_buffer = self.current_agent.model.rollout_buffer
        policy = self.current_agent.model.policy
        
        ''' Update Variables for the current Model's Obersvations and Actions '''
        _, wk_check, wk_ckm = self.MoveGenerator.GenerateLegalMoves(self.board.white_king, self.board)
        _, bk_check, bk_ckm = self.MoveGenerator.GenerateLegalMoves(self.board.black_king, self.board)
        self.board.update_board_state(wk_check, wk_ckm, bk_check, bk_ckm)
        
        while True:
            # self.draw_to_screen()
            if rollout_buffer.full:
                with th.no_grad():
                # Compute value for the last timestep
                    values = policy.predict_values(obs_as_tensor(self.board.get_state(self.current_agent.team), self.current_agent.model.device))
                    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=np.array([0]))
                
                print('Training Agent...')
                self.current_agent.model.train()
                rollout_buffer.reset()

            policy.set_training_mode(False)

            # Sample new weights for the state dependent exploration
            if self.current_agent.model.use_sde:
                policy.reset_noise(1)


            if self.current_agent.model.use_sde and self.current_agent.model.sde_sample_freq > 0:
                    # Sample a new noise matrix
                    policy.reset_noise(1)
            obs = self.board.get_state(self.current_agent.team)
            
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(obs, self.current_agent.model.device)
                actions, values, log_probs = policy(obs_tensor)
                
            actions = actions.cpu().numpy()
            # Rescale and perform action
            clipped_actions = actions

            # Clip the actions to avoid out of bound error
            if isinstance(self.current_agent.model.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.current_agent.model.action_space.low, self.current_agent.model.action_space.high)
            

            piece = self.board.get_square((clipped_actions[0,0], clipped_actions[0,1]))
            move = (clipped_actions[0,2], clipped_actions[0,3])
            moveset, _, _ = self.MoveGenerator.GenerateLegalMoves(piece, self.board)

            _, rewards, dones, infos = self.current_agent.env.step(piece, move, moveset, obs['check'])
            # self.draw_to_screen()
            # Give access to local variables
            # time.sleep(0.1)
            if isinstance(self.current_agent.model.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            
            rollout_buffer.add(obs, actions, rewards, np.ones(shape=(1,)), values, log_probs)
            if dones:
                print('Game Finished...')
                self._perform_losing_agents_pass_through()
            elif infos['valid_move']:
                break


        

                        
    def train_agents(self):
        ''' Number of Game Sets '''
        for _ in range(self.num_epochs):
            ''' Number of Games in each Set '''
            for _ in range(self.num_games):
                ''' Initialize the Game '''
                self.agent1.env.reset()
                self.agent2.env.reset()
                while not any(self.board.get_winner(self.Piece.Color.BLACK)):
                    self.collect_agent_rollout_buffer()
                    time.sleep(0.1)
                    self.current_agent = self.current_agent.next
                    
                status = self.board.get_winner()
                '''                  Agent 1     Agent 2 '''
                self.score_line += status[1] - status[2]
                print('Current Scoreline: ' + str(self.score_line))
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
                
                            
