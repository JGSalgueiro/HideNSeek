import pygame
from game.agent import Agent
from game.constants import HIDE, SEEK, WIDTH, HEIGHT
from game.board import Board

FPS = 1

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Hide and Seek')

def main():
    run = True
    clock = pygame.time.Clock()
    board = Board()
    hidingAgents = [Agent(0, 1, HIDE), Agent(0, 2, HIDE), Agent(4, 6, HIDE)]
    seekingAgents = [Agent(9, 3, SEEK), Agent(7, 2, SEEK), Agent(9, 6, SEEK)]

    while run:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        board.draw_cells(WIN)
        for agent in hidingAgents:
            agent.draw(WIN)

        for agent in seekingAgents:
            agent.draw(WIN)

        pygame.display.update()
            

    pygame.quit()


main()