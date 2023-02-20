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
        self.max_speed = 100
        self.min_speed = 0
        self.reverse_speed = -3
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, keys, dt):
        
        # Turn left when the left arrow key is pressed
        if keys[pygame.K_LEFT]:
            self.turn_left()
        # Turn right when the right arrow key is pressed
        elif keys[pygame.K_RIGHT]:
            self.turn_right()
        else:
            self.angle_change = 0

        # Accelerate when the up arrow key is pressed
        if keys[pygame.K_UP]:
            self.accelerate()
        # Brake when the down arrow key is pressed
        elif keys[pygame.K_DOWN]:
            self.brake()
        else:
            self.speed *= 0.95

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

        # Update the self.rect object to reflect the car's new position and size
        self.rect.x, self.rect.y = self.x - self.width/2, self.y - self.height/2
        self.rect.width, self.rect.height = self.width, self.height

    def turn_left(self):
        self.angle_change = self.turning_speed

    def turn_right(self):
        self.angle_change = -self.turning_speed

    def accelerate(self):
        self.speed += 1
        if self.speed > self.max_speed:
            self.speed = self.max_speed

    def brake(self):
        if self.speed > self.min_speed:
            self.speed -= 1
        else:
            self.speed = self.reverse_speed

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

