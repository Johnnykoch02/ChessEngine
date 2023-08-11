NSEW = 0
DIAGANOL = 1
PAWN_WHITE = 2
PAWN_BLACK = 3
HORSIE = 4
KING = 5

from ..Utils.imports import Piece, Board, add_two_pos, is_in_board
class MoveGenerator:
    @staticmethod
    def GenerateLegalMoves(piece, board):
        """
        Figure out which piece it is
        """
        if piece is None:
            print('Piece is none')
            return set(), None, None
        MoveTypes = Move.GetMoveSet(piece)
        LegalMoves = []
        PseudoLegalMoves = set()
        king_in_check = False
        
        v_board = board.create_virtual_board()
        v_piece = Piece.create_virtual_piece(piece)
        if v_board.get_square(v_piece.square) is None:
            v_board.play_move(v_piece, v_piece.square)        
        if not MoveGenerator.is_king_in_check(piece.color, v_board):
            MoveGenerator.GetPseudoLegalMoves(PseudoLegalMoves, MoveTypes, piece, board, False)
        else:
            king_in_check = True
        
        MoveGenerator.GetPseudoLegalMoves(PseudoLegalMoves, MoveTypes, piece, board)
            
        # print('PseudoLegalMoves', PseudoLegalMoves)
            
        for move in PseudoLegalMoves:
            virtual_board = board.create_virtual_board() 
            vPiece = Piece.create_virtual_piece(piece)
            virtual_board.play_move(vPiece, move)
            
            if not MoveGenerator.is_king_in_check(vPiece.color, virtual_board):
                LegalMoves.append(move)
            else:
                pass

        king_in_checkmate = False
        if king_in_check:
            if not MoveGenerator.any_move_removes_check(board, piece):
                king_in_checkmate = True

        return LegalMoves, king_in_check, king_in_checkmate
    
    @staticmethod
    def any_move_removes_check(board, piece):
        pieces = None       
        if piece.color == Piece.Color.WHITE:
            pieces = board.get_white_pieces()
        elif piece.color == Piece.Color.BLACK:
            pieces = board.get_black_pieces()
            
        for piece in pieces:
            virtual_board = board.create_virtual_board() 
            vPiece = virtual_board.get_square(piece.square)
            PseudoLegalMoves = set()
            MoveGenerator.GetPseudoLegalMoves(PseudoLegalMoves, Move.GetMoveSet(vPiece), vPiece, virtual_board)
            for move in PseudoLegalMoves:
                virtual_board2 = virtual_board.create_virtual_board() 
                vPiece2 = virtual_board2.get_square(piece.square)
                virtual_board2.play_move(vPiece2, move)
                if not MoveGenerator.is_king_in_check(piece.color, virtual_board2):
                    return True
        return False

    @staticmethod
    def is_king_in_check(current_color, board):
        if current_color == Piece.Color.WHITE:
            black_pieces = board.get_black_pieces()
            
            for piece in black_pieces:
                virtual_board = board.create_virtual_board() 
                vPiece = virtual_board.get_square(piece.square)
                PseudoLegalMoves = set()
                MoveGenerator.GetPseudoLegalMoves(PseudoLegalMoves, Move.GetMoveSet(vPiece), vPiece, virtual_board)
                for move in PseudoLegalMoves:
                    if virtual_board.white_king and move == virtual_board.white_king.square:
                        # print('King in Check')
                        return True

        elif current_color == Piece.Color.BLACK:
            white_pieces = board.get_white_pieces()
            
            for piece in white_pieces:
                virtual_board = board.create_virtual_board() 
                vPiece = virtual_board.get_square(piece.square)
                PseudoLegalMoves = set()
                MoveGenerator.GetPseudoLegalMoves(PseudoLegalMoves, Move.GetMoveSet(vPiece), vPiece, virtual_board)
                for move in PseudoLegalMoves:
                    if virtual_board.black_king and move == virtual_board.black_king.square:
                        # print('King in Check')
                        return True
                        
        return False

    @staticmethod
    def GetPseudoLegalMoves(PseudoLegalMoves, MoveTypes, piece, board, debug=False):
        initial_pos = piece.square
        for move_type in MoveTypes:
            if move_type == NSEW:
                n = (-1, 0)
                s = (1,0)
                e = (0,1)
                w = (0,-1)
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, n), board, n, debug) # North
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, s), board, s, debug) # South
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, e), board, e, debug) # East
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, w), board, w, debug) # West
            if move_type == DIAGANOL:
                ne = (1, -1)
                se = (1, 1)
                nw = (-1,-1)
                sw = (-1, 1)
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, ne), board, ne, debug) # NE
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, se), board, se, debug) #SE
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, nw), board, nw, debug) #NW
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, sw), board, sw, debug) #SW
            if move_type == PAWN_WHITE:
                MoveGenerator.pawn_moves(PseudoLegalMoves, piece, initial_pos, board, 1)
            if move_type == PAWN_BLACK:
                MoveGenerator.pawn_moves(PseudoLegalMoves, piece, initial_pos, board, -1)
            if move_type == KING:
                MoveGenerator.king_moves(PseudoLegalMoves, piece, initial_pos, board)
            if move_type == HORSIE:
                MoveGenerator.horsie_moves(PseudoLegalMoves, piece, initial_pos, board)

    @staticmethod
    def king_moves(PseudoLegalMoves, piece, pos, board):
        poses = add_two_pos(pos, (-1, -1)), add_two_pos(pos, (-1,0)), add_two_pos(pos, (-1,1)), add_two_pos(pos, (0, 1)), add_two_pos(pos, (0,-1)), add_two_pos(pos, (1,0)), add_two_pos(pos, (1,-1)), add_two_pos(pos, (1,1)) 
        for spot in poses:
            if is_in_board(spot):
                temp_piece = board.get_square(spot)
                if temp_piece is None:
                    PseudoLegalMoves.add(spot)
                else:
                    if temp_piece.color != piece.color:
                        PseudoLegalMoves.add(spot)
    @staticmethod
    def horsie_moves(PseudoLegalMoves, piece, pos, board):
        poses = add_two_pos(pos, (-1, -2)), add_two_pos(pos, (-1,2)), add_two_pos(pos, (1,-2)), add_two_pos(pos, (1, 2)), add_two_pos(pos, (2,-1)), add_two_pos(pos, (2,1)), add_two_pos(pos, (-2,-1)), add_two_pos(pos, (-2,1)) 
        for spot in poses:
            if is_in_board(spot):
                temp_piece = board.get_square(spot)
                if temp_piece is None:
                    PseudoLegalMoves.add(spot)
                else:
                    if temp_piece.color != piece.color:
                        PseudoLegalMoves.add(spot)    
        
        '''
            - Finish the King  /
            - Finish Horsie Logic /
            DONE FOR PSEUDOLEGAL
            - take the pseudolegal moves, play the move on a virtual board, check to see if the king is in check
            - if the king is not in check, then the move is valid

            Move Generation:
            - Check if the king is in check
                -if the king is in check, then you need to only be able to play those moves.
                -if the king is not in check, then generate all the pseudolegal moves.
                - Play all of the pseudolegal moves that we just made, and get the legal moves.

        '''

    @staticmethod
    def pawn_moves(PsuedoLegalMoves, piece, pos, board, direction):
        if direction == 1 and pos[0] == 1: #White Piece
            #Beginning Piece
            temp_piece = board.get_square(add_two_pos(pos, (2,0)))
            if temp_piece is None:
                PsuedoLegalMoves.add(add_two_pos(pos, (2,0)))
        
        if direction == -1 and pos[0] == 6: # Black Piece
            #Beginning Piece
            temp_piece = board.get_square(add_two_pos(pos, (-2,0)))
            if temp_piece is None:
                PsuedoLegalMoves.add(add_two_pos(pos, (-2,0)))

        # Straight Forward  
        temp_piece = board.get_square(add_two_pos(pos, (direction,0)))
        if temp_piece is None:
            PsuedoLegalMoves.add(add_two_pos(pos, (direction, 0)))
        # 1 Diagonal    
        temp_piece = board.get_square(add_two_pos(pos, (direction, direction)))
        if temp_piece is not None and piece.color != temp_piece.color:
            PsuedoLegalMoves.add(add_two_pos(pos, (direction, direction)))
        # 1 Diagonal
        temp_piece = board.get_square(add_two_pos(pos, (direction, -direction)))
        if temp_piece is not None and piece.color != temp_piece.color:
            PsuedoLegalMoves.add(add_two_pos(pos, (direction, -direction)))                   
                
    @staticmethod
    def MoveInDirectionNorm(PsuedoLegalMoves, piece, pos, board, direction, debug=False):
        if not is_in_board(pos):
            return
        temp_piece = board.get_square(pos)
        if temp_piece is None:
            PsuedoLegalMoves.add(pos)
            MoveGenerator.MoveInDirectionNorm(PsuedoLegalMoves, piece, add_two_pos(pos, direction), board, direction, True)
        else:
            if temp_piece.color != piece.color:
                PsuedoLegalMoves.add(pos)
            else:
                if debug:
                    pass
                    # print('BLOCKED: by',temp_piece.color != piece.color, 'Piece:', temp_piece.type, temp_piece.square )
            return

class Move:
    @staticmethod
    def GetMoveSet(piece:'Piece'):
        MoveTypes = []
        if piece.type == Piece.Type.BISHOP or piece.type == Piece.Type.QUEEN:
            MoveTypes.append(DIAGANOL)
        if piece.type == Piece.Type.QUEEN or piece.type == Piece.Type.ROOK:
            MoveTypes.append(NSEW)
        elif piece.type == Piece.Type.KNIGHT:
            MoveTypes.append(HORSIE)
        elif piece.type == Piece.Type.KING:
            MoveTypes.append(KING)
        elif piece.type == Piece.Type.PAWN:
            MoveTypes.append(PAWN_WHITE) if piece.color == Piece.Color.WHITE else MoveTypes.append(PAWN_BLACK)
        
        return MoveTypes