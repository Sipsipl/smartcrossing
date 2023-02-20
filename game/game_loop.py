import pygame
from game.car import Car
from game.obstacle import Obstacle
from game.road import Road

# Define the game constants
WIDTH = 640
HEIGHT = 480
FPS = 30

# Initialize Pygame
pygame.init()

# Set up the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smart Crossing")

# Set up the game clock
clock = pygame.time.Clock()

# Create the game entities
car = Car(x=0, y=0, width=50, height=50, color=(255, 0, 0))
obstacle = Obstacle(x=100, y=100, width=50, height=50, color=(0, 255, 0))
road = Road(x=0, y=300, width=600, height=50, color=(128, 128, 128))

# Game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Quit the game
            pygame.quit()
            exit()

    # Update game state
    car.update()
    obstacle.update()
    road.update()

    # Render graphics
    screen.fill((255, 255, 255))
    road.draw(screen)
    obstacle.draw(screen)
    car.draw(screen)

    # Update the display
    pygame.display.update()

    # Wait for next frame
    clock.tick(FPS)
