import math
import pygame
from game.config import WIDTH, HEIGHT

class Car:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.angle = 0
        self.speed = 0
        self.turning_speed = 30

    def update(self, keys, dt):
        # Calculate the car's speed and angle based on the arrow keys
        if keys[pygame.K_UP]:
            self.speed += 1
        elif keys[pygame.K_DOWN]:
            self.speed -= 1
        else:
            self.speed *= 0.95

        # Calculate the change in angle based on the arrow keys
        if keys[pygame.K_LEFT]:
            self.angle_change = self.turning_speed
        elif keys[pygame.K_RIGHT]:
            self.angle_change = -self.turning_speed
        else:
            self.angle_change = 0

        # Gradually update the angle
        self.angle += self.angle_change * dt

        # Convert the angle to radians
        angle_radians = math.radians(self.angle)

        # Update the car's position based on its speed and angle
        new_x = self.x + self.speed * math.cos(angle_radians) * dt
        new_y = self.y - self.speed * math.sin(angle_radians) * dt

        # Check if the new position is within the screen boundaries
        if new_x - self.width / 2 < 0:
            new_x = self.width / 2
        elif new_x + self.width / 2 > WIDTH:
            new_x = WIDTH - self.width / 2

        if new_y - self.height / 2 < 0:
            new_y = self.height / 2
        elif new_y + self.height / 2 > HEIGHT:
            new_y = HEIGHT - self.height / 2

        self.x, self.y = new_x, new_y

    def draw(self, surface):
        # Create a new surface for the car
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Draw the car
        rect = pygame.draw.rect(car_surface, self.color, (0, 0, self.width, self.height))
        pygame.draw.circle(car_surface, (255, 255, 255), (int(self.width / 4), self.height), int(self.height / 4))
        pygame.draw.circle(car_surface, (255, 255, 255), (int(self.width * 3 / 4), self.height), int(self.height / 4))

        # Rotate the car surface
        rotated_car_surface = pygame.transform.rotate(car_surface, self.angle)

        # Calculate the position of the car surface
        rect = rotated_car_surface.get_rect()
        rect.center = (self.x, self.y)

        # Draw the car surface to the screen
        surface.blit(rotated_car_surface, rect)
