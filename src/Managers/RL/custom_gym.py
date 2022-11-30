'''
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

            rew+= (Own Team Score - MAX_SCORE ) - (Ot Team Score - MAX_SCORE)

        if WIN:
            rew+=250
        
        if LOSE:
            rew+= -250
        
        if DRAW rew+= -5
'''
import os
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, Box, Dict, Space
from math import inf, radians, degrees

import time
#import AppManager

VERSION = 'GM-PPO-0.0.1'

CHECKPOINT_DIR = os.path.join(os.getcwd(), 'Reinforcement Learning','src','RL','Training', 'Checkpoints', NAME)
LOG_DIR = os.path.join(os.getcwd(), 'Training', 'Logs', VERSION)
SAVE_FREQ = 200


class GrandMasterEnv(Env):
    from ...Utils.imports import Piece, Board
    def __init__(self, board:Board =None, team_color:Piece.Color = None):
        super(GrandMasterEnv, self).__init__()
        '''Class Variables Required for Env'''
        self._board = board
        self._team_color = team_color

        spaces = {
            'board_state': Box(low=0, high=(6,2), shape=(8,8,2)),
            'team_color': Discrete(low=1, high=2),
            'score': Box(low=-inf, high=inf)
        }


    
    def set_board(self, board):
        self._board = board
    
    def set_team_color(self, team_color):
        self._team_color = team_color
    