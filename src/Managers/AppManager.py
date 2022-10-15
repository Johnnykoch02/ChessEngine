
from collections import deque
from ..Utils.imports import pygame, SQUARE_DIMENSIONS, BACKROUND_COLOR, APP_DIMENSIONS

game_screen = [None]
drawables = []

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

        ## 

        # Initialize Application Paramaters
        pygame.display.set_caption('Chess: by John')
        self.screen.fill(self.background_color)
        self.Running = True

        # from ..Lib.piece import Piece
        # Piece(Piece.Type.KNIGHT,Piece.Color.BLACK, (5,5))
        ##

        from src.Utils.imports import Board 
        self.Board = Board
        self.board = self.Board()

        self.current_player = Player('black')
        self.current_player.next = Player('white')
        self.piece_selected = None
        # Start the App

        self.Run()
    
    def Run(self):
        i = 0
        while self.Running:
            ## Top of Game Loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.Running = False

            # Mouse Events
            if i % 3 == 0:
                i = 0
                mouse_pos = pygame.mouse.get_pos()
                presses = pygame.mouse.get_pressed()
                square_on = (mouse_pos[1]//SQUARE_DIMENSIONS[0], mouse_pos[0]//SQUARE_DIMENSIONS[0]) # Mouse to Square
                if presses[0]:
                   if self.piece_selected:
                    pq = self.board.get_square(square_on)
                    if pq:
                        pq.destroy()
                    self.piece_selected.square = square_on
                    self.piece_selected.selected = False
                    self.piece_selected = None
                   else:
                        self.piece_selected = self.board.get_square(square_on)
                        if self.piece_selected:
                            self.piece_selected.selected = True
                
                if self.piece_selected:
                    self.piece_selected.set_screen_pos(mouse_pos)
                     
            i+=1
            self.Draw()

    def Draw(self):
        self.screen.fill(self.background_color)
        for drawable in drawables:
            drawable.Draw()
        pygame.display.flip()

