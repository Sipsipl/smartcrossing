import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game.game_loop import GameLoop, GameState
import pygame
import traceback
from .logger import setup_logger

# Set up logger
logger = setup_logger("smartcrossing.env")

class SmartCrossingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Initialize the game
        self.game = None
        self.render_mode = render_mode
        self.clock = None
        
        # Initialize pygame if we're rendering
        if self.render_mode == "human":
            try:
                logger.info("Initializing pygame for rendering")
                if not pygame.get_init():
                    pygame.init()
                self.clock = pygame.time.Clock()
            except Exception as e:
                logger.error(f"Failed to initialize pygame: {str(e)}")
                logger.debug(traceback.format_exc())
                raise
        
        # Define action space (Discrete: 0=nothing, 1=left, 2=right, 3=accelerate, 4=brake)
        self.action_space = spaces.Discrete(5)
        
        # Define observation space
        # [x, y, angle, speed, goal_x, goal_y, nearest_obstacle_x, nearest_obstacle_y]
        low = np.array([0, 0, -360, -3, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1024, 768, 360, 150, 1024, 768, 1024, 768], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        logger.debug("Environment initialized successfully")

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
            
            logger.debug("Resetting environment")
            # Reset the game to initial state
            self.game = GameLoop()
            
            # Get initial observation
            observation = self._get_obs()
            info = self._get_info()
            
            # Render initial state if needed
            if self.render_mode == "human":
                try:
                    self.render()
                except Exception as e:
                    logger.error(f"Failed to render initial state: {str(e)}")
                    logger.debug(traceback.format_exc())
                    raise
            
            return observation, info
        
        except Exception as e:
            logger.error(f"Failed to reset environment: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def step(self, action):
        try:
            # Map actions to game controls
            keys = {pygame.K_LEFT: False, pygame.K_RIGHT: False, 
                    pygame.K_UP: False, pygame.K_DOWN: False}
                    
            if action == 1:  # Left
                keys[pygame.K_LEFT] = True
            elif action == 2:  # Right
                keys[pygame.K_RIGHT] = True
            elif action == 3:  # Accelerate
                keys[pygame.K_UP] = True
            elif action == 4:  # Brake
                keys[pygame.K_DOWN] = True
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None, 0, True, True, {}
            
            # Update game state
            if self.render_mode == "human":
                dt = self.clock.tick(60) / 1000.0  # Use actual time delta
            else:
                dt = 1.0 / 60.0  # Fixed timestep for non-rendering mode
            
            # Update car based on current action
            self.game.car.update(keys, dt)
            
            # Update game state
            self.game.update(dt)
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Check if episode is done
            terminated = self._is_terminated()
            truncated = False
            
            # Get observation and info
            observation = self._get_obs()
            info = self._get_info()
            
            # Render if needed
            if self.render_mode == "human":
                try:
                    self.render()
                    pygame.event.pump()  # Process events to keep the window responsive
                except Exception as e:
                    logger.error(f"Failed to render step: {str(e)}")
                    logger.debug(traceback.format_exc())
                    raise
            
            return observation, reward, terminated, truncated, info
        
        except Exception as e:
            logger.error(f"Failed to execute step with action {action}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def _get_obs(self):
        """Convert game state to observation array"""
        try:
            car = self.game.car
            goal = self.game.goal
            obstacle = self.game.obstacle
            
            return np.array([
                car.rect.centerx,
                car.rect.centery,
                car.angle,
                car.speed,
                goal.rect.centerx,
                goal.rect.centery,
                obstacle.rect.centerx,
                obstacle.rect.centery,
            ], dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Failed to get observation: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def _get_info(self):
        """Get additional information about the environment state"""
        try:
            car = self.game.car
            goal = self.game.goal
            
            # Calculate distance to goal using rect centers
            distance_to_goal = np.sqrt(
                (car.rect.centerx - goal.rect.centerx) ** 2 + 
                (car.rect.centery - goal.rect.centery) ** 2
            )
            
            return {
                "distance_to_goal": distance_to_goal
            }
        
        except Exception as e:
            logger.error(f"Failed to get info: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def _calculate_reward(self):
        """Calculate the reward based on current state"""
        try:
            reward = 0
            car = self.game.car
            goal = self.game.goal
            obstacle = self.game.obstacle
            
            # Reward for reaching goal (using GameState)
            if self.game.state.goals_reached > getattr(self, '_previous_goals', 0):
                reward += 100
                self._previous_goals = self.game.state.goals_reached
            
            # Penalty for collision with obstacle
            if car.check_collision(obstacle):
                reward -= 50
            
            # Small penalty for time
            reward -= 0.1
            
            # Reward for getting closer to goal (based on previous distance)
            current_distance = np.sqrt(
                (car.rect.centerx - goal.rect.centerx) ** 2 + 
                (car.rect.centery - goal.rect.centery) ** 2
            )
            if hasattr(self, '_previous_distance'):
                reward += (self._previous_distance - current_distance) * 0.1
            self._previous_distance = current_distance
            
            return reward
        
        except Exception as e:
            logger.error(f"Failed to calculate reward: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def _is_terminated(self):
        """Check if episode should end"""
        try:
            car = self.game.car
            goal = self.game.goal
            obstacle = self.game.obstacle
            
            # End if car reaches goal or hits obstacle
            return (car.check_collision(goal) or 
                    car.check_collision(obstacle))
        
        except Exception as e:
            logger.error(f"Failed to check termination: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def render(self):
        """Render the environment"""
        try:
            if self.render_mode == "human":
                if pygame.get_init():
                    self.game.render()
                    pygame.event.pump()  # Keep the window responsive
                else:
                    logger.warning("Pygame not initialized, skipping render")
        except Exception as e:
            logger.error(f"Failed to render environment: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def close(self):
        """Clean up resources"""
        try:
            if self.render_mode == "human" and pygame.get_init():
                pygame.quit()
        except Exception as e:
            logger.error(f"Failed to close environment: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
