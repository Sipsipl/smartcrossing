from typing import List, Optional
import pygame
from game.car import Car
from game.obstacle import Obstacle
from game.road import Road
from game.goal import Goal
from game.game_objects import GameObject, DynamicGameObject, StaticGameObject
from game.config import *

class GameState:
    """Represents the current state of the game"""
    def __init__(self):
        self.running: bool = True
        self.score: int = 0
        self.time: float = 0
        self.goals_reached: int = 0

class GameLoop:
    """Main game loop handling updates and rendering"""
    def __init__(self):
        """Initialize the game"""
        # Initialize Pygame
        pygame.init()

        # Set up the game window
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Smart Crossing")

        # Set up the game clock
        self.clock = pygame.time.Clock()
        
        # Initialize game state
        self.state = GameState()
        
        # Create game objects
        self._create_game_objects()

        # Initialize goal collision state
        self._goal_reached = False

    def _create_game_objects(self) -> None:
        """Create and initialize all game objects"""
        # Dynamic objects (can move)
        self.dynamic_objects: List[DynamicGameObject] = [
            Car(x=0, y=0, width=60, height=30, color=(255, 0, 0))
        ]
        self.car = self.dynamic_objects[0]  # Main player car
        
        # Static objects (don't move)
        self.static_objects: List[StaticGameObject] = [
            Obstacle(x=160, y=160, width=80, height=80, color=(0, 0, 0)),
            Road(x=0, y=480, width=1024, height=80, color=(128, 128, 128)),
            Goal(x=800, y=480, width=80, height=80, color=(0, 255, 0))  # Made goal square for perfect circle
        ]
        
        # Store specific static objects for easy access
        self.obstacle = self.static_objects[0]
        self.road = self.static_objects[1]
        self.goal = self.static_objects[2]

    def handle_events(self) -> None:
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state.running = False

    def update(self, dt: float) -> None:
        """Update game state
        
        Args:
            dt: Time delta since last update in seconds
        """
        # Get input
        keys = pygame.key.get_pressed()
        
        # Update dynamic objects
        for obj in self.dynamic_objects:
            if isinstance(obj, Car):
                obj.update(keys, dt)
            else:
                obj.update(dt)
        
        # Update static objects
        for obj in self.static_objects:
            obj.update()
        
        # Check collisions
        self._handle_collisions()
        
        # Update game state
        self.state.time += dt

    def _handle_collisions(self) -> None:
        """Handle collisions between game objects"""
        # Check car collision with goal
        if self.car.check_collision(self.goal) and not self._goal_reached:
            print("Goal achieved!")
            self.state.goals_reached += 1
            self.state.score += 100
            self._goal_reached = True
        elif not self.car.check_collision(self.goal):
            self._goal_reached = False
        
        # Check car collision with obstacle
        if self.car.check_collision(self.obstacle):
            self.state.score -= 50

    def render(self) -> None:
        """Render the game state"""
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Draw static objects first
        for obj in self.static_objects:
            obj.draw(self.screen)
        
        # Draw dynamic objects on top
        for obj in self.dynamic_objects:
            obj.draw(self.screen)
        
        # Update display
        pygame.display.update()

    def run(self) -> None:
        """Main game loop"""
        while self.state.running:
            # Handle events
            self.handle_events()
            
            # Update game state
            dt = self.clock.tick(FPS) / 1000.0
            self.update(dt)
            
            # Render graphics
            self.render()

        # Cleanup
        pygame.quit()
        exit()
