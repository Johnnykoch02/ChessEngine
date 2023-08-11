'''
    total number of possible states:

    approx. (8*8*2*6) * (2) * (500) * (2) \~= 1,536,000 different possible Observation States
    Board       Team  Score  Check

    Piece   Move

    Num_Observation_States * Num_Action_States = 3,687,936,000 possible Q States

    Therefore, Q Learning is not an Optimal RL Algorithm for this problem.

    This envirornment will have two PPO 
    Observation Space:
        board:
            is of (8x8) with the potential of piece or no piece. 
            pieces of type [1-6] and of color [1-2], input channels = 2
            [8 x 8 image with 2 channels]

        Team Color: 
            Discrete Value: [1-2]
            
        
    Action Space:
        Square of Piece:
            MultiDiscrete Value: [0-7]
            (xPos, yPos)
        
        Square to Move:
            MultiDiscrete Value: [0-7]
            (xPos, yPos)

    Reward Function:

        if move is in Piece's Legal Moveset:
            rew+=1
        else:
            rew+=-10
            return

        Team Color Dependent:
            (Calculate Piece Deprecation)
            Something like a dot product piece type and piece value:
                Vector DP column of piece values or 0-if piece is captured-
                multiplied by the number of pieces associated with those values
                the MAX is if no pieces have been caputred

            rew+= (Own Team Score - Ot Team Score)

        if WIN:
            rew+=250
        
        if LOSE:
            rew+= -250
        
        if DRAW rew+= -5
'''
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, Box, Dict, Space
from math import inf, radians, degrees
import numpy as np
import torch as th
import time

observation_space = {
            'board_state': Box(low=0, high=6, shape=(3,8,8), dtype=np.float32),
            'team_color': Box(low=0, high=1, shape=(1,),dtype=np.float32),
            'score': Box(low=-10, high=10, shape=(1,),dtype=np.float32),
            'check': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            # 'piece_to_move': Box(low=0, high=64, shape=(1,),dtype=np.float32)
        }

action_space = Discrete(64)
# action_space = MultiDiscrete([64, 64])


class GrandMasterEnv(Env):
    global observation_space, action_space
    def __init__(self, board =None, team_color = None):
        from ...Utils.imports import MoveGenerator
        super(GrandMasterEnv, self).__init__()
        '''Class Variables Required for Env'''
        self._board = board
        self._team_color = team_color
        self.MoveGen = MoveGenerator
        

        self.observation_space = Dict(observation_space)
        
        self.state = None    
        if self._board:
            self.state = self._board.get_state(self._team_color)

        self.action_space = action_space
    
    def set_board(self, board):
        self._board = board
    
    def set_team_color(self, team_color):
        self._team_color = team_color

    def step(self, piece, move, moveset, check):
        '''
        '''
        info = {'valid_move': False}
        if piece is None:
            pass
        elif piece.color is not self._team_color:
            pass
        elif move not in moveset:
            pass
        else:
            info['valid_move'] = True
            self._board.play_move(piece, move)
            
        time.sleep(0.1)
        state = self._board.get_state(self._team_color)
        reward, done = self.get_currrent_reward(state['score'], check)
        
        return state, reward, done, info


    def reset(self):
        if not self._board.is_reset():
            self._board.reset_board()

    def get_currrent_reward(self, score: float, check: bool):
        done = False
        reward = 0
        winner = self._board.get_winner(self._team_color)

        if self._board.moves_played > 300:
            done = True
            reward = score
            return reward, done
            
        if winner[0]:
            done = True
            reward= 5* int(winner[1]) - 5 * int(winner[2]) 
            return reward, done
        
        '''  Valid Move     US           Them  '''
        reward+= check[1] - check[0] + 0.1 * score
        
        return reward, done
    
    @staticmethod
    def get_sim_reward(Board, team_color):
        obs = Board.get_state(team_color)
        done = False
        reward = 0
        winner = Board.get_winner(team_color)
            
        if winner[0]:
            done = True
            reward= 5 * int(winner[1]) - 5 * int(winner[2]) 
            return reward, done
        
        '''  Valid Move     US           Them  '''
        reward = obs['check'][1] - obs['check'][0] + 0.1 * obs['score'][0]     
        
        return reward, done

