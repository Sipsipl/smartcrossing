import pygame
from game.game_objects import StaticGameObject

class Goal(StaticGameObject):
    """Goal that the car needs to reach"""
    def __init__(self, x: float, y: float, width: float, height: float, color: tuple[int, int, int]):
        # Make width and height equal for a perfect circle
        size = min(width, height)
        super().__init__(x, y, size, size, color, shape='circle')