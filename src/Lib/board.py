import numpy as np
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
        self.wK_in_check = False
        self.wK_in_checkmate = False
        self.bK_in_check = False
        self.bK_in_checkmate = False

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
         
        if piece.type == piece.Type.PAWN:
            if piece.square[0] == 0 and piece.color == piece.Color.BLACK or piece.square[0] == 7 and piece.color == piece.Color.WHITE:
                piece.type = piece.Type.QUEEN
                
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
                    pygame.draw.rect(game_screen[0], SQUARE_COLOR, (i*width, j*height, width, height))
        
        for pos in self.selected_squares:
            # pygame.draw.rect(game_screen[0], SELECTED_COLOR, (pos[1]*width, pos[0]*height, width, height))
            s = pygame.Surface((width,height))  # the size of your rect
            s.set_alpha(50)                # alpha level
            s.fill(SELECTED_COLOR)           # this fills the entire surface
            game_screen[0].blit(s, (pos[1]*width,pos[0]*height))    # (0,0) are the top-left coordinates
    
    def get_state(self, team_color):
        board_state = np.zeros(shape=(8,8,2))
        for piece in self.pieces:
            board_state[piece.square[0], piece.square[1], 0] = int(piece.color) + 1
            board_state[piece.square[0], piece.square[1], 1] = int(piece.type) + 1
        white_score = self.get_white_score()
        black_score = self.get_black_score()
        us = False
        them = False
        if team_color == self.PieceClass.Color.WHITE:
            us = self.wK_in_check
            them = self.bK_in_check
        else:
            us = self.bK_in_check
            them = self.wK_in_check
        
        score = white_score - black_score if team_color == self.PieceClass.Color.WHITE else black_score - white_score

        return {
            'board_state': board_state,
            'team_color': int(team_color) + 1,
            'score': score,
            'check': np.array([us, them])
        }

    def update_board_state(self, white_king_check, white_king_checkmate, black_king_check, black_king_checkmate):
        self.wK_in_check = white_king_check
        self.wK_in_checkmate = white_king_checkmate
        self.bK_in_check = black_king_check
        self.bK_in_checkmate = black_king_checkmate

    def get_white_score(self):
        white_score = 0
        for piece in self.pieces:
            if piece.color == piece.Color.WHITE:
                white_score += piece.score()
        return white_score
        
    def get_black_score(self):
        black_score = 0
        for piece in self.pieces:
            if piece.color == piece.Color.BLACK:
                black_score += piece.score()
        return black_score
    
    def get_winner(self, team_color):
        if self.bK_in_checkmate or self.wK_in_checkmate:
            win = False
            if team_color == self.PieceClass.Color.WHITE:
                return [win, self.bK_in_checkmate, self.wK_in_checkmate]
            else:
                return [win, self.wK_in_checkmate, self.bK_in_checkmate]
        return [False, False]

    def reset_board(self):
        for piece in self.pieces:
            piece.destroy()

        self.pieces = []

        self.selected_squares = []
        self.current_state = START_POS
        self.pieces = self.init_board()


        self.white_king = None
        self.black_king = None
        self.wK_in_check = False
        self.wK_in_checkmate = False
        self.bK_in_check = False
        self.bK_in_checkmate = False

        for piece in self.pieces:
            if piece.type == piece.Type.KING:
                if piece.color == piece.Color.WHITE:
                    self.white_king = piece 
                else:
                    self.black_king = piece
        self.current_color = None