#app manager

from .config import BACKROUND_COLOR, SQUARE_COLOR, SELECTED_COLOR, APP_DIMENSIONS, SQUARE_DIMENSIONS, add_two_pos, is_in_board

# board
import pygame as pygame

#pieces 

import os

from main import PROJECT_PATH

from ..Managers.AppManager import drawables, game_screen, spritesheet, AppManager

from ..Lib.board import Board

from ..Lib.piece import Piece, get_piece_from_fen, get_sprite_from_piece

from ..Lib.MoveGenerator import MoveGenerator

from ..Managers.RL.GrandMasterPPO import GrandMasterPPO, GrandMasterJudge, AgentPtr

