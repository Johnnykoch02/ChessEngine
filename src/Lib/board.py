from ..Managers.AppManager import drawables, game_screen
from ..Utils.config import APP_DIMENSIONS, SQUARE_COLOR
import pygame

FILE = 8
RANK = 8

START_POS = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'



class Board:

    def __init__(self):
        drawables.append(self)
        self.current_state = START_POS

    def Draw(self):
        width = APP_DIMENSIONS[0]//RANK
        height = APP_DIMENSIONS[1]//FILE
        
        for i in range(FILE):
            for j in range(i%2, RANK, 2):
                pygame.draw.rect(game_screen[0], SQUARE_COLOR, (i*width, j*height, width, height))

        

