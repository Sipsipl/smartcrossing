import pygame
from game.game_objects import StaticGameObject

class Road(StaticGameObject):
    """The road that the car drives on"""
    def __init__(self, x: float, y: float, width: float, height: float, color: tuple[int, int, int]):
        super().__init__(x, y, width, height, color)