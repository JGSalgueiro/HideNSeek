import pygame
import random
from .constants import GREEN, HIDE, RED, SQUARE_SIZE, WHITE

class Agent:
    PADDING = 10
    BORDER = 2

    def __init__(self, row, col, team):
        self.row = row
        self.col = col
        self.team = team
        self.eliminated = False
        self.direction = None

        if self.team == HIDE:
            self.color = GREEN
        else:
            self.color = RED
        self.x = 0
        self.y = 0
        self.calc_position()

    def calc_position(self):
        self.x = SQUARE_SIZE * self.col + SQUARE_SIZE // 2
        self.y = SQUARE_SIZE * self.row + SQUARE_SIZE // 2

    def eliminate(self):
        self.eliminate = True
    
    def move(self):
        decision = random.randint(0,3)
        print(decision)

        if decision == 0:
            self.col = self.col - 1

        elif decision == 1:
            self.row = self.row + 1

        elif decision == 2:
            self.col = self.col + 1

        elif decision == 3:
            self.row = self.row - 1

    def draw(self, win):
        self.move()
        radius = SQUARE_SIZE // 2 - self.PADDING
        pygame.draw.circle(win, WHITE, (self.x, self.y), radius + self.BORDER)
        pygame.draw.circle(win, self.color, (self.x, self.y), radius)

    def get_team(self):
        return self.team

    def get_position(self):
        return self.x, self.y

    def __repr__(self):
        return str(self.team)