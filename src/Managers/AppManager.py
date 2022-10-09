
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
        # Start the App

        self.Run()
    
    def Run(self):
        while self.Running:
            ## Top of Game Loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.Running = False

            self.Draw()

    def Draw(self):
        self.screen.fill(self.background_color)
        for drawable in drawables:
            drawable.Draw()
        pygame.display.flip()

