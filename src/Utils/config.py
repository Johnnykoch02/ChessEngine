### Change basic Configurartion File
BACKROUND_COLOR = (65, 105, 225) # Royal Blue
SQUARE_COLOR = (224, 247, 250) # Really white blue
SELECTED_COLOR = (244,54,76)

APP_DIMENSIONS = (656, 656)
SQUARE_DIMENSIONS = (APP_DIMENSIONS[0]/8, APP_DIMENSIONS[1]/8)



def add_two_pos(pos1, pos2):
    return (pos1[0] + pos2[0], pos1[1] + pos2[1])

def is_in_board(pos):
    return (pos[0] >= 0 and pos[0] < 8 and pos[1] >= 0 and pos[1] < 8)