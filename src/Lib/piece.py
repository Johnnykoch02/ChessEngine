from enum import Enum
import os
from ..Managers.AppManager import drawables, game_screen, spritesheet
from ..Utils.config import SQUARE_DIMENSIONS
import pygame

''''rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'''
# spritesheet = spritesheet(os.path.join(project_path, 'data', 'assets', 'pieces.png'), 1, 1)

class Piece:
    
    class Type(Enum): ## Characteristic Type
        KING = 0
        QUEEN = 1
        KNIGHT = 2
        BISHOP = 3
        ROOK = 4
        PAWN = 5

    class Color(Enum):
        WHITE = 0
        BLACK = 1
    
    def __init__(self, type:Type, color:Color, square):
        self.type = type
        self.color = color
        self.square = square
        drawables.append(self)
        
    
    def Draw(self):                                     
        game_screen[0].blit(get_sprite_from_piece(self),
             (SQUARE_DIMENSIONS * self.square[0], SQUARE_DIMENSIONS*self.square[1]))
    
     # Till next time.


def get_piece_from_fen(character:str, square):
    color = Piece.Color.WHITE if character.islower() else Piece.Color.BLACK
    character = character.lower()
    type = {'r':Piece.Type.ROOK, 'n':Piece.Type.KNIGHT,
             'b': Piece.Type.BISHOP, 'q':Piece.Type.QUEEN,
              'k':Piece.Type.KING, 'p':Piece.Type.PAWN}[character]
    return Piece(type, color, square)
    

def get_sprite_from_piece(piece:'Piece'):
    if piece.color == Piece.Color.WHITE:
        return {Piece.Type.QUEEN: (0,0,333,333), Piece.Type.KING: (333,0,333,333), 
        Piece.Type.BISHOP:(666,0,333,333), Piece.Type.KNIGHT: (999,0,333,333), 
        Piece.Type.ROOK: (1332,0,333,333), Piece.Type.PAWN:(1665,0,333,333)}[piece.type]
    else:
         return {Piece.Type.QUEEN: (0,334,333,333), Piece.Type.KING: (333,334,333,333), 
        Piece.Type.BISHOP:(666,334,333,333), Piece.Type.KNIGHT: (999,334,333,333), 
        Piece.Type.ROOK: (1332,334,333,333), Piece.Type.PAWN:(1665,334,333,333)}[piece.type]