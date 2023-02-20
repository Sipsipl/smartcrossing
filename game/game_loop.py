import pygame
from game.car import Car
from game.obstacle import Obstacle
from game.road import Road
from game.config import *

# Initialize Pygame
pygame.init()

# Set up the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smart Crossing")

# Set up the game clock
clock = pygame.time.Clock()

# Create the game entities
car = Car(x=0, y=0, width=100, height=50, color=(255, 0, 0))
obstacle = Obstacle(x=100, y=100, width=50, height=50, color=(0, 255, 0))
road = Road(x=0, y=300, width=600, height=50, color=(128, 128, 128))

# Game loop
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
    dt = clock.tick(FPS) / 1000.0
    car.update(keys, dt)
    obstacle.update()
    road.update()

    # Render graphics
    screen.fill((255, 255, 255))
    road.draw(screen)
    obstacle.draw(screen)
    car.draw(screen)

    # Update the display
    pygame.display.update()

# Quit the game
pygame.quit()
exit()

