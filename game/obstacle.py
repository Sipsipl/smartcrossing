import pygame
from game.game_objects import StaticGameObject

# obstacle.py
class Obstacle(StaticGameObject):
    """An obstacle that the car needs to avoid"""
    def __init__(self, x: float, y: float, width: float, height: float, color: tuple[int, int, int]):
        super().__init__(x, y, width, height, color)