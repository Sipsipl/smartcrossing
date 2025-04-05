import os
from typing import List, Optional
import pygame
from game.car import Car
from game.obstacle import Obstacle
from game.road import Road
from game.goal import Goal
from game.game_objects import DynamicGameObject, StaticGameObject
import traceback
import logging

# Set up logger
logger = logging.getLogger("smartcrossing.game")

# Constants
WIDTH = 1024
HEIGHT = 768
FPS = 60

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
        try:
            # Initialize Pygame only if not already initialized
            if not pygame.get_init():
                pygame.init()
            
            # Initialize display module if not already initialized
            if not pygame.display.get_init():
                pygame.display.init()
            
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
            
            logger.debug("Game loop initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize game loop: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def _create_game_objects(self) -> None:
        """Create and initialize all game objects"""
        try:
            # Create dynamic objects (car)
            self.dynamic_objects: List[DynamicGameObject] = [
                Car(x=100, y=480, width=60, height=30, color=(255, 0, 0))  # Start car on the road
            ]
            
            # Create static objects (obstacle, road, goal)
            self.static_objects: List[StaticGameObject] = [
                Obstacle(x=160, y=160, width=80, height=80, color=(0, 0, 0)),
                Road(x=0, y=480, width=1024, height=80, color=(128, 128, 128)),
                Goal(x=800, y=480, width=80, height=80, color=(0, 255, 0))
            ]
            
            # Store specific static objects for easy access
            self.obstacle = self.static_objects[0]
            self.road = self.static_objects[1]
            self.goal = self.static_objects[2]
            
            # Store car for easy access
            self.car = self.dynamic_objects[0]
            
            logger.debug("Game objects created successfully")
        
        except Exception as e:
            logger.error(f"Failed to create game objects: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def handle_events(self) -> None:
        """Handle pygame events"""
        try:
            # Check if Pygame and display are initialized
            if not pygame.get_init() or not pygame.display.get_init():
                logger.warning("Pygame or display not initialized, reinitializing...")
                if not pygame.get_init():
                    pygame.init()
                if not pygame.display.get_init():
                    pygame.display.init()
                # Recreate the screen if needed
                if not hasattr(self, 'screen') or self.screen is None:
                    self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                    pygame.display.set_caption("Smart Crossing")
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state.running = False
                    logger.info("Received quit event")
                    # Don't quit pygame here, let the run method handle cleanup
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.state.running = False
                    logger.info("Received escape key press")
                    # Don't quit pygame here, let the run method handle cleanup
            
            # Keep the window responsive
            pygame.event.pump()
        
        except Exception as e:
            logger.error(f"Failed to handle events: {str(e)}")
            logger.debug(traceback.format_exc())
            # Don't raise the exception, just log it and continue
            # This prevents the game from crashing due to event handling issues

    def update(self, dt: float) -> None:
        """Update game state
        
        Args:
            dt: Time delta since last update in seconds
        """
        try:
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
        
        except Exception as e:
            logger.error(f"Failed to update game state: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def _handle_collisions(self) -> None:
        """Handle collisions between game objects"""
        try:
            # Check car collision with goal
            if self.car.check_collision(self.goal) and not self._goal_reached:
                logger.info("Goal achieved!")
                self.state.goals_reached += 1
                self.state.score += 100
                self._goal_reached = True
            elif not self.car.check_collision(self.goal):
                self._goal_reached = False
            
            # Check car collision with obstacle
            if self.car.check_collision(self.obstacle):
                self.state.score -= 50
                logger.debug("Collision with obstacle")
                # Reset car position instead of ending the game
                self.car.reset_position()
                # Optional: Add a small delay or visual feedback here
        
        except Exception as e:
            logger.error(f"Failed to handle collisions: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def render(self) -> None:
        """Render the game state"""
        try:
            # Check if Pygame and display are initialized
            if not pygame.get_init() or not pygame.display.get_init():
                logger.warning("Pygame or display not initialized, reinitializing...")
                if not pygame.get_init():
                    pygame.init()
                if not pygame.display.get_init():
                    pygame.display.init()
                # Recreate the screen if needed
                if not hasattr(self, 'screen') or self.screen is None:
                    self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                    pygame.display.set_caption("Smart Crossing")
            
            # Clear screen
            self.screen.fill((255, 255, 255))
            
            # Draw static objects first
            for obj in self.static_objects:
                obj.draw(self.screen)
            
            # Draw dynamic objects on top
            for obj in self.dynamic_objects:
                obj.draw(self.screen)
            
            # Update display
            pygame.display.flip()  # Use flip() instead of update() for double buffering
            pygame.event.pump()  # Keep the window responsive
            
            logger.debug("Frame rendered successfully")
        
        except Exception as e:
            logger.error(f"Failed to render game state: {str(e)}")
            logger.debug(traceback.format_exc())
            # Don't raise the exception, just log it and continue
            # This prevents the game from crashing due to render issues

    def run(self) -> None:
        """Main game loop"""
        try:
            while self.state.running:
                # Handle events
                self.handle_events()
                
                # Update game state
                dt = self.clock.tick(FPS) / 1000.0
                self.update(dt)
                
                # Render graphics
                self.render()
            
            logger.info("Game loop ended normally")
        
        except Exception as e:
            logger.error(f"Game loop crashed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
        
        finally:
            # Cleanup
            try:
                if pygame.display.get_init():
                    pygame.display.quit()
                if pygame.get_init():
                    pygame.quit()
                logger.info("Pygame resources cleaned up")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                logger.debug(traceback.format_exc())
