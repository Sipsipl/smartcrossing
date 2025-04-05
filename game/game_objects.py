from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pygame
import math

class Drawable(ABC):
    """Interface for objects that can be drawn on screen"""
    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the object on the screen"""
        pass

class Collidable(ABC):
    """Interface for objects that can collide with other objects"""
    @abstractmethod
    def check_collision(self, other: 'Collidable') -> bool:
        """Check if this object collides with another object"""
        pass

class Movable(ABC):
    """Interface for objects that can move"""
    @abstractmethod
    def move(self, dx: float, dy: float) -> None:
        """Move the object by the given delta"""
        pass

class GameObject(Drawable, Collidable, ABC):
    """Base class for all game objects"""
    def __init__(self, x: float, y: float, width: float, height: float, color: Tuple[int, int, int]):
        """Initialize a game object
        
        Args:
            x: X coordinate
            y: Y coordinate
            width: Width of the object
            height: Height of the object
            color: RGB color tuple
        """
        self.color = color
        self.rect = pygame.Rect(x, y, width, height)
        # Store dimensions for later use
        self.width = width
        self.height = height

    def check_collision(self, other: 'GameObject') -> bool:
        """Check if this object collides with another object"""
        return self.rect.colliderect(other.rect)

    def update(self) -> None:
        """Update the object state"""
        pass

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the object
        
        Args:
            screen: Surface to draw on
        """
        pygame.draw.rect(screen, self.color, self.rect)

class StaticGameObject(GameObject):
    """Base class for non-moving game objects"""
    def __init__(self, x: float, y: float, width: float, height: float, color: Tuple[int, int, int], shape: str = 'rect'):
        super().__init__(x, y, width, height, color)
        self.shape = shape

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the object with its specified shape
        
        Args:
            screen: Surface to draw on
        """
        if self.shape == 'circle':
            # For circles, we draw centered in the rect with diameter = min(width, height)
            diameter = min(self.width, self.height)
            center = self.rect.center
            pygame.draw.circle(screen, self.color, center, diameter / 2)
        else:
            # Default to rectangle
            pygame.draw.rect(screen, self.color, self.rect)

class DynamicGameObject(GameObject, Movable):
    """Base class for moving game objects"""
    def __init__(self, x: float, y: float, width: float, height: float, color: Tuple[int, int, int]):
        super().__init__(x, y, width, height, color)
        self.angle = 0.0
        self.velocity = pygame.math.Vector2(0, 0)
        # Create a surface for rotation
        self._original_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(self._original_surface, color, (0, 0, width, height))

    def move(self, dx: float, dy: float) -> None:
        """Move the object by the given delta"""
        self.rect.x += dx
        self.rect.y += dy

    def rotate(self, angle: float) -> None:
        """Rotate the object by the given angle"""
        self.angle = (self.angle + angle) % 360
        # Rotate the surface
        rotated_surface = pygame.transform.rotate(self._original_surface, self.angle)
        # Get the new rect and center it
        self.rect = rotated_surface.get_rect(center=self.rect.center)

    def update(self, dt: float) -> None:
        """Update the object state
        
        Args:
            dt: Time delta since last update
        """
        # Apply velocity
        self.move(self.velocity.x * dt, self.velocity.y * dt)

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the rotated object
        
        Args:
            screen: Surface to draw on
        """
        # Create rotated surface
        rotated_surface = pygame.transform.rotate(self._original_surface, self.angle)
        # Get rect for positioning
        draw_rect = rotated_surface.get_rect(center=self.rect.center)
        # Draw to screen
        screen.blit(rotated_surface, draw_rect)
