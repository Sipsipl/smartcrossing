import math
import pygame
from game.config import WIDTH, HEIGHT
from game.game_objects import DynamicGameObject

class Car(DynamicGameObject):
    """A car that can be controlled by the player or AI"""
    def __init__(self, x: float, y: float, width: float, height: float, color: tuple[int, int, int]):
        super().__init__(x, y, width, height, color)
        # Treat the provided coordinates as the car's centre so that dynamic
        # positioning logic (which works with centre coordinates) remains
        # consistent even after resets or collision responses.
        self.rect.center = (x, y)

        # Store initial centre position for reset
        self.initial_x = self.rect.centerx
        self.initial_y = self.rect.centery

        # Position tracking (needed for accurate movement)
        self.x = float(self.rect.centerx)
        self.y = float(self.rect.centery)
        # Car properties
        self.turning_speed = 40
        self.max_speed = 150
        self.min_speed = 0
        self.reverse_speed = -3
        self.speed = 0
        self.angle_change = 0

    def update(self, keys: dict, dt: float) -> None:
        """Update the car's state based on input and time delta
        
        Args:
            keys: Dictionary of keyboard state
            dt: Time delta since last update
        """
        # Turn left when the left arrow key is pressed
        if keys[pygame.K_LEFT]:
            self.angle_change = self.turning_speed
        # Turn right when the right arrow key is pressed
        elif keys[pygame.K_RIGHT]:
            self.angle_change = -self.turning_speed
        else:
            self.angle_change = 0

        # Accelerate when the up arrow key is pressed
        if keys[pygame.K_UP]:
            self.speed += 1
            if self.speed > self.max_speed:
                self.speed = self.max_speed
        # Brake when the down arrow key is pressed
        elif keys[pygame.K_DOWN]:
            if self.speed > self.min_speed:
                self.speed -= 1
            else:
                self.speed = self.reverse_speed
        else:
            self.speed *= 0.95

        # Update angle
        self.rotate(self.angle_change * dt)

        # Convert angle to radians
        angle_radians = math.radians(self.angle)

        # Update position based on speed and angle
        self.x += self.speed * math.cos(angle_radians) * dt
        self.y -= self.speed * math.sin(angle_radians) * dt

        # Check boundaries
        if self.x - self.rect.width/2 < 0:
            self.x = self.rect.width/2
        elif self.x + self.rect.width/2 > WIDTH:
            self.x = WIDTH - self.rect.width/2

        if self.y - self.rect.height/2 < 0:
            self.y = self.rect.height/2
        elif self.y + self.rect.height/2 > HEIGHT:
            self.y = HEIGHT - self.rect.height/2

        # Update rect position
        self.rect.centerx = self.x
        self.rect.centery = self.y

    def reset_position(self) -> None:
        """Reset the car to its initial position and state"""
        # Reset position
        self.x = self.initial_x
        self.y = self.initial_y
        
        # Reset speed and angle
        self.speed = 0
        self.angle = 0
        
        # Update rect position
        self.rect.centerx = self.x
        self.rect.centery = self.y
