import pygame
import sys
import os
from configs import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from jogo import Game

# Add the parent directory to sys.path to import menu
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from menu import Menu

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Casino Clássico - Blackjack")
    clock = pygame.time.Clock()

    # Show menu
    menu = Menu(screen)
    action = menu.run()
    if action == "start_game":
        game = Game(screen)
        while True:
            events = pygame.event.get()
            game.handle_events(events)
            game.render()
            clock.tick(FPS)

if __name__ == "__main__":
    main()
