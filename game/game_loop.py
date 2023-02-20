# game_loop.py
import pygame
from game.car import Car
from game.obstacle import Obstacle
from game.road import Road
from game.goal import Goal
from game.config import *

class GameLoop:
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Set up the game window
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Smart Crossing")

        # Set up the game clock
        self.clock = pygame.time.Clock()

        # Create the game entities
        self.car = Car(x=0, y=0, width=100, height=50, color=(255, 0, 0))
        self.obstacle = Obstacle(x=100, y=100, width=50, height=50, color=(0, 0, 0))
        self.road = Road(x=0, y=300, width=600, height=50, color=(128, 128, 128))
        self.goal = Goal(x=500, y=300, width=100, height=50, color=(0, 255, 0))

    def run(self):
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Quit the game
                    pygame.quit()
                    exit()

            # Get the current state of the arrow keys
            keys = pygame.key.get_pressed()

            # Update game state
            dt = self.clock.tick(FPS) / 1000.0
            self.car.update(keys, dt)
            self.obstacle.update()
            self.road.update()
            self.goal.update()

            # Check for collisions with the goal
            if self.car.rect.colliderect(self.goal.rect):
                print("Goal achieved!")

            # Render graphics
            self.screen.fill((255, 255, 255))
            self.road.draw(self.screen)
            self.obstacle.draw(self.screen)
            self.goal.draw(self.screen)
            self.car.draw(self.screen)

            # Update the display
            pygame.display.update()

        # Quit the game
        pygame.quit()
        exit()
