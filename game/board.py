import pygame
from .constants import BLACK, LIGHT_BLUE, ROWS, SQUARE_SIZE

class Board:
    def __init__(self):
        self.board= []
        self.turn = 0
        self.playerTurn = None
        self.hideTeam = self.seekTeam = 3

    def draw_cells(self, win):
        win.fill(BLACK)
        for row in range(ROWS):
            for col in range(row % 2, ROWS, 2):
                pygame.draw.rect(win, LIGHT_BLUE, (row*SQUARE_SIZE, col*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def create_board(self):
        for row in range(ROWS):
            self.append()
        pass
