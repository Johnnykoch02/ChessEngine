
from collections import deque
from ..Utils.imports import pygame, SQUARE_DIMENSIONS, BACKROUND_COLOR, APP_DIMENSIONS, os
import threading
import time

game_screen = [None]
drawables = []

VERSION = 'GM-PPO-0.0.0'
LOG_DIR = os.path.join(os.getcwd(), 'src','Managers','RL', 'Logs', VERSION)
NAME = "GrandMaster_v_0_0_0"
CHECKPOINT_DIR = os.path.join(os.getcwd(), 'src','Managers','RL', 'Checkpoints', NAME)

project_path = ''

class spritesheet:
    def __init__(self, path, rows: int, cols: int):
        self.path = path
        self.rows = rows
        self.cols = cols
        self.img = pygame.image.load(self.path).convert_alpha()

    ## Rectangle: (x, y, width, height)
    def get(self, rect):
        rectangle = pygame.Rect(rect)
        image = pygame.Surface(rectangle.size, pygame.SRCALPHA)
        image.blit(self.img, (0,0), rectangle)
        image = pygame.transform.scale(image, SQUARE_DIMENSIONS)
        return image

class Player:
    def __init__(self, type:'str'):
        from src.Utils.imports import Piece
        self.PieceTypes = Piece.Color
        self.type = self.PieceTypes.BLACK if type == 'black' else self.PieceTypes.WHITE
        self.move_stack = deque()
        self.next = None

class AppManager:
    def __init__(self, path):
        global project_path   
        
        # Initialize all required params
        self.background_color = BACKROUND_COLOR
        self.screen = pygame.display.set_mode(APP_DIMENSIONS)
        game_screen[0] = self.screen
        project_path = path


        # Initialize Application Paramaters
        pygame.display.set_caption('Chess: by John')
        self.screen.fill(self.background_color)
        self.Running = True

        self.mode = 'default'
        self.judge = None

        from src.Utils.imports import Board, MoveGenerator
        self.Board = Board
        self.board = self.Board()

        self.current_player = Player('black')
        self.current_player.next = Player('white')
        self.current_player.next.next = self.current_player
        self.piece_selected = None
        self.legal_moves = None

        self.board.current_color = self.current_player.type
        # Start the App
        self.MoveGenerator = MoveGenerator
        
        self.draw_thread = threading.Thread(target=self.DrawCaller)
        self.judge_thread = None
        self.draw_thread.daemon = True
    
    def Run(self):
        i = 0
        last_mouse_press = None
        while self.Running:
                ## Top of Game Loop
            if self.mode != 'train':
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.Running = False

                # Mouse Events
                if i % 5 == 0:
                    i = 0
                    presses = pygame.mouse.get_pressed()
                    mouse_pos = pygame.mouse.get_pos()
                    if presses != last_mouse_press:
                        self.mouse_events(mouse_pos, presses)

                    last_mouse_press = presses   
                    if self.piece_selected:
                        self.piece_selected.set_screen_pos(mouse_pos)

                i+=1
                # time.sleep(0.2)
                self.Draw()

    def mouse_events(self, mouse_pos, presses): 
        square_on = (int(mouse_pos[1]//SQUARE_DIMENSIONS[0]), int(mouse_pos[0]//SQUARE_DIMENSIONS[0])) # Mouse to Square
        print(square_on)
        if presses[0]:
           if self.piece_selected:
            made_move_on_board = not (self.piece_selected.square == square_on)
            if made_move_on_board:
                if square_on in self.legal_moves:
                    self.board.play_move(self.piece_selected, square_on)
                    self.piece_selected.selected = False
                    self.piece_selected = None
                    self.board.selected_squares = []
                    self.current_player = self.current_player.next  
                    self.board.current_color = self.current_player.type 
            else:
                self.board.play_move(self.piece_selected, square_on)
                self.piece_selected.selected = False
                self.board.selected_squares = []
                self.piece_selected = None

           else: #We need to validate the proper color has been selected
                temp_piece = self.board.get_square(square_on)
                if temp_piece is not None and temp_piece.color == self.current_player.type:
                    self.piece_selected = temp_piece
                if self.piece_selected:
                    self.board.remove_piece(self.piece_selected)
                    self.piece_selected.selected = True
                    pseudolegal, _, _ = self.MoveGenerator.GenerateLegalMoves(self.piece_selected, self.board)
                    self.legal_moves = tuple(tuple([int(x[0]), int(x[1])]) for x in pseudolegal)
                    self.board.selected_squares = self.legal_moves
                    print(self.board.selected_squares)
    
    
    def Train(self):
        from src.Utils.imports import GrandMasterPPO, GrandMasterJudge, AgentPtr
        global CHECKPOINT_DIR, LOG_DIR
        self.judge = GrandMasterJudge(
            AgentPtr(GrandMasterPPO(tensorboard_log=LOG_DIR)),
            AgentPtr(GrandMasterPPO(tensorboard_log=LOG_DIR)),
            CHECKPOINT_DIR, LOG_DIR,
            self.Draw     
        )
        self.mode = 'train'
        
        self.draw_thread.start()
        self.judge_thread = threading.Thread(target=self.judge.train_agents)
        # self.judge_thread.daemon = True
        # self.judge_thread.start()
        self.judge.train_agents()
        self.Run()

    def DrawCaller(self):
        while True:
            self.Draw()
            time.sleep(0.2)
            
    def Draw(self):
        self.screen.fill(self.background_color)
        for drawable in drawables:
            drawable.Draw()
        pygame.display.flip()



# def mouse_events(self, mouse_pos, presses): 
#         square_on = (mouse_pos[1]//SQUARE_DIMENSIONS[0], mouse_pos[0]//SQUARE_DIMENSIONS[0]) # Mouse to Square
#         print(square_on)
#         if presses[0]:
#            if self.piece_selected:
#             pq = self.board.get_square(square_on) # Query the board
#             if pq and self.piece_selected.color != pq.color or not pq:
#                 if self.piece_selected != pq and pq:
#                     self.board.remove_piece(pq)
#                     pq.destroy()
#                 made_move_on_board = not (self.piece_selected.square == square_on)
#                 self.piece_selected.square = square_on
#                 self.piece_selected.selected = False
#                 self.board.place_piece(self.piece_selected)
#                 self.piece_selected = None
#                 if made_move_on_board:
#                     self.current_player = self.current_player.next  
#                     self.board.current_color = self.current_player.type                  
#            else: #We need to validate the proper color has been selected
#                 temp_piece = self.board.get_square(square_on)
#                 if temp_piece is not None and temp_piece.color == self.current_player.type:
#                     self.piece_selected = temp_piece
#                 if self.piece_selected:
#                     self.board.remove_piece(self.piece_selected)
#                     self.piece_selected.selected = True
