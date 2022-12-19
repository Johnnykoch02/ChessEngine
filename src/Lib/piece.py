from enum import IntEnum
from ..Utils.imports import PROJECT_PATH, spritesheet, drawables, SQUARE_DIMENSIONS,game_screen, os

''''rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'''
# This is our Piece Sprite Sheet
sp = None

class Piece:
    
    class Type(IntEnum): ## Characteristic Type
        KING = 0
        QUEEN = 1
        KNIGHT = 2
        BISHOP = 3
        ROOK = 4
        PAWN = 5

    class Color(IntEnum):
        WHITE = 0
        BLACK = 1
    
    def __init__(self, type:Type, color:Color, square, visible=True):
        self.type = type
        self.visible = visible
        self.color = color
        self.square = square
        self.selected = False
        self.screen_pos = None
        if visible:
            drawables.append(self)
    
    @staticmethod
    def create_virtual_piece(piece:'Piece'):
        return Piece(piece.type, piece.color, piece.square, visible=False)

    def set_screen_pos(self, pos):
        self.screen_pos = pos
        self.square_on = None    
    
    def score(self):
        return{Piece.Type.PAWN:1 + self.distance_from_queen(),
         Piece.Type.KNIGHT:4, Piece.Type.ROOK:5, Piece.Type.BISHOP:5,
         Piece.Type.QUEEN: 10, Piece.Type.KING: 20}[self.type]

    def destroy(self):
        self.selected = False
        if self.visible:
            drawables.remove(self)
    
    def distance_from_queen(self):
        if self.color == Piece.Color.WHITE:
            return (7 - self.square[0]) / 7
        elif self.color == Piece.Color.BLACK:
            return self.square[0] / 7    

    def Draw(self):  
        global sp
        if sp == None:
            sp = spritesheet(os.path.join(PROJECT_PATH, 'data', 'assets', 'pieces.png'), 1, 1)
        if not self.selected:       
            YPOS = SQUARE_DIMENSIONS[0] * self.square[0]
            XPOS = SQUARE_DIMENSIONS[1]*self.square[1]

            game_screen[0].blit(sp.get(get_sprite_from_piece(self)),
               (XPOS, YPOS) )
        else:
            game_screen[0].blit(sp.get(get_sprite_from_piece(self)),
               self.screen_pos )
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
        return {Piece.Type.KING: (0,0,333,333), Piece.Type.QUEEN: (333,0,333,333), 
        Piece.Type.BISHOP:(666,0,333,333), Piece.Type.KNIGHT: (999,0,333,333), 
        Piece.Type.ROOK: (1332,0,333,333), Piece.Type.PAWN:(1665,0,333,333)}[piece.type]
    else:
         return {Piece.Type.KING: (0,334,333,333), Piece.Type.QUEEN: (333,334,333,333), 
        Piece.Type.BISHOP:(666,334,333,333), Piece.Type.KNIGHT: (999,334,333,333), 
        Piece.Type.ROOK: (1332,334,333,333), Piece.Type.PAWN:(1665,334,333,333)}[piece.type]