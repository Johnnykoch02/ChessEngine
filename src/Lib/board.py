
from ..Utils.imports import  SQUARE_COLOR, SELECTED_COLOR, APP_DIMENSIONS, drawables, pygame, game_screen
FILE = 8
RANK = 8

START_POS = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'



class Board:
    def __init__(self, virtual=False):
        from ..Utils.imports import get_piece_from_fen, Piece
        self.get_piece_from_fen = get_piece_from_fen
        self.PieceClass = Piece
        self.pieces = []
        if virtual:
            return
        drawables.append(self)
        self.selected_squares = []
        self.current_state = START_POS
        self.pieces = self.init_board()


        self.white_king = None
        self.black_king = None
        for piece in self.pieces:
            if piece.type == piece.Type.KING:
                if piece.color == piece.Color.WHITE:
                    self.white_king = piece 
                else:
                    self.black_king = piece
        self.current_color = None

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

    def create_virtual_board(self):
        virtual_board = Board(True)
        for piece in self.pieces:
            virtual_board.pieces.append(self.PieceClass.create_virtual_piece(piece))
        virtual_board.white_king = None
        virtual_board.black_king = None
        for piece in virtual_board.pieces:
            if piece.type == piece.Type.KING:
                if piece.color == piece.Color.WHITE:
                    virtual_board.white_king = piece 
                else:
                    virtual_board.black_king = piece
        virtual_board.current_color = self.current_color
        return virtual_board
    
    def remove_piece(self, piece):
        for p in self.pieces:
            if piece == p:
                self.pieces.remove(piece)
                break
            
    def place_piece(self, piece):
        self.pieces.append(piece)
    
    def play_move(self, piece, move):
        self.remove_piece(piece)
        pq = self.get_square(move)
        if pq is not None:
            if pq == self.white_king:
                self.white_king.piece = None
            elif pq == self.black_king:
                self.black_king.piece = None
            self.remove_piece(pq)
            pq.destroy()
        piece.square = move
        self.place_piece(piece)
    
    def get_black_pieces(self):
        black_pieces = []
        for piece in self.pieces:
            if piece.color == piece.Color.BLACK:
                black_pieces.append(piece)
        return black_pieces

    def get_white_pieces(self):
        white_pieces = []
        for piece in self.pieces:
            if piece.color == piece.Color.WHITE:
                white_pieces.append(piece)
        return white_pieces

    def Draw(self):
        width = APP_DIMENSIONS[0]//RANK
        height = APP_DIMENSIONS[1]//FILE
        
        for i in range(FILE):
            for j in range(i%2, RANK, 2):
                # if (j, i) in self.selected_squares:
                #     pygame.draw.rect(game_screen[0], SELECTED_COLOR, (i*width, j*height, width, height))
                # else:
                    pygame.draw.rect(game_screen[0], SQUARE_COLOR, (i*width, j*height, width, height))

