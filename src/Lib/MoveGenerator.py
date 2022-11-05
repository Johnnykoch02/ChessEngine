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
                n, s, e, w = (0, -1), (0,1), (1,0), (-1,0)
                MoveGenerator.MoveInDirection(PseudoLegalMoves, piece, add_two_pos(initial_pos, n), board, n) # North
                MoveGenerator.MoveInDirection(PseudoLegalMoves, piece, add_two_pos(initial_pos, s), board, s) #South
                MoveGenerator.MoveInDirection(PseudoLegalMoves, piece, add_two_pos(initial_pos, e), board, e) #East
                MoveGenerator.MoveInDirection(PseudoLegalMoves, piece, add_two_pos(initial_pos, w), board, w) #West
            

    @staticmethod
    def MoveInDirection(PsuedoLegalMoves, piece, pos, board, direction):
        if not is_in_board(pos):
            return
        temp_piece = board.get_square(pos)
        if temp_piece is None:
            PsuedoLegalMoves.add(pos)
            MoveGenerator.MoveInDirection(PsuedoLegalMoves, piece, add_two_pos(pos, direction), board, direction)
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

        
