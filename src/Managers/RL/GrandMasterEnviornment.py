'''
    total number of possible states:

    (8*8*2*6) * (2) * (500) * (2) = 1,536,000 different possible Observation States
    Board       Team  Score  Check

    (7*7) * (7*7) = 2401 different Possible Action States
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
import time
#import AppManager

observation_space = {
            'board_state': Box(low=0, high=6, shape=(8,8,2)),
            'team_color': Discrete(1),
            'score': Box(low=-250, high=250, shape=(1,)),
            'check': MultiDiscrete([1,1])
        }

action_space = MultiDiscrete([7, 7, 7, 7])


class GrandMasterEnv(Env):
    global observation_space, action_space
    from ...Utils.imports import Piece, Board
    def __init__(self, board:Board =None, team_color:Piece.Color = None):
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
        if not piece is None or piece.color is not self._team_color or move not in moveset:
            info['valid_move'] = True
            self._board.play_move(piece, move)
        time.sleep(0.5)
        state = self._board.get_state(self._team_color)
        reward, done = self.get_currrent_reward(state['score'], piece, move, moveset, check)
        
        return state, reward, done, info


    def reset(self):
        if not self._board.is_reset():
            self._board.reset_board()

    def get_currrent_reward(self, score, piece, move, moveset, check):
        done = False
        reward = 0
        winner = self._board.get_winner(self._team_color)
        if winner[0]:
            done = True
            reward= 250* int(winner[1]) - 250 * int(winner[2]) 
            return reward, done

        if piece is None:
            reward = -25
            return reward, done
        if piece.color is not self._team_color:
            reward = -25
            return reward, done
        if move not in moveset:
            reward = -25
            return reward, done
        
        '''  Valid Move     US           Them  '''
        reward+= 1 + 50*check[1] - 50*check[0]        
        
        return reward + 1.8 * score
        
        

