NSEW = 0
DIAGANOL = 1
PAWN_WHITE = 2
PAWN_BLACK = 3
HORSIE = 4
KING = 5

from ..Utils.imports import Piece, add_two_pos, is_in_board
class MoveGenerator:
    @staticmethod
    def GenerateLegalMoves(piece, board):
        """
        Figure out which piece it is
        """
        MoveTypes = Move.GetMoveSet(piece)
        LegalMoves = []
        PseudoLegalMoves = {}
        MoveGenerator.GetPseudoLegalMoves(PseudoLegalMoves, MoveTypes, piece, board)

    @staticmethod
    def GetPseudoLegalMoves(PseudoLegalMoves, MoveTypes, piece, board):
        initial_pos = piece.square
        for move_type in MoveTypes:
            if move_type == NSEW:
                n, s, e, w = (-1, 0), (1,0), (0,1), (0,-1)
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, n), board, n) # North
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, s), board, s) # South
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, e), board, e) # East
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, w), board, w) # West
            if move_type == DIAGANOL:
                ne, se, nw, sw = (1, -1), (1, 1), (-1,-1), (1,-1)
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, n), board, ne) # NE
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, s), board, se) #SE
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, e), board, nw) #NW
                MoveGenerator.MoveInDirectionNorm(PseudoLegalMoves, piece, add_two_pos(initial_pos, w), board, sw) #SW
            if move_type == PAWN_WHITE:
                MoveGenerator.pawn_moves(PseudoLegalMoves, piece, initial_pos, board, 1)
            if move_type == PAWN_BLACK:
                MoveGenerator.pawn_moves(PseudoLegalMoves, piece, initial_pos, board, -1)
            if move_type == KING:
                MoveGenerator.king_moves(PseudoLegalMoves, piece, initial_pos, board)

    @staticmethod
    def king_moves(PseudoLegalMoves, piece, pos, board):
        poses = add_two_pos(pos, (-1, -1)), add_two_pos(pos, (-1,0)), add_two_pos(pos, (-1,1)), add_two_pos(pos, (0, 1)), add_two_pos(pos, (0,-1)), add_two_pos(pos, (1,0)), add_two_pos(pos, (1,-1)), add_two_pos(pos, (1,1)) 
        valid_poses = []
        for spot in poses:
            if is_in_board(spot):
                valid_poses.append(spot)
        '''
            - Finish the King Logic
            - Finish Horsie Logic
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
        if temp_piece is not None:
            PsuedoLegalMoves.add(add_two_pos(pos, (direction, direction)))
        # 1 Diagonal
        temp_piece = board.get_square(add_two_pos(pos, (direction, -direction)))
        if temp_piece is not None:
            PsuedoLegalMoves.add(add_two_pos(pos, (direction, -direction)))                   
                
    @staticmethod
    def MoveInDirectionNorm(PsuedoLegalMoves, piece, pos, board, direction):
        if not is_in_board(pos):
            return
        temp_piece = board.get_square(pos)
        if temp_piece is None:
            PsuedoLegalMoves.add(pos)
            MoveGenerator.MoveInDirectionNorm(PsuedoLegalMoves, piece, add_two_pos(pos, direction), board, direction)
        else:
            if temp_piece.color != piece.color:
                PsuedoLegalMoves.add(pos)
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

        
