
from ..Utils.imports import  SQUARE_COLOR, APP_DIMENSIONS, drawables, pygame, game_screen
FILE = 8
RANK = 8

START_POS = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'



class Board:
    def __init__(self):
        from ..Utils.imports import get_piece_from_fen
        self.get_piece_from_fen = get_piece_from_fen
        drawables.append(self)
        self.current_state = START_POS
        self.pieces = self.init_board()

    def init_board(self):
         
        pieces = []
        row = 0
        col = 0
        for c in START_POS:
            if c == '/':
                row+=1
                col=0
                continue
            try:
                spaces= int(c) # assuming this will cause an error
                col+=spaces
            except:
                pieces.append(self.get_piece_from_fen(c, (row, col)))
                col+=1
        
        return pieces

    def get_square(self, pos):
        for piece in self.pieces:
            if piece.square == pos:
                return piece
        return None

    def Draw(self):
        width = APP_DIMENSIONS[0]//RANK
        height = APP_DIMENSIONS[1]//FILE
        
        for i in range(FILE):
            for j in range(i%2, RANK, 2):
                pygame.draw.rect(game_screen[0], SQUARE_COLOR, (i*width, j*height, width, height))

