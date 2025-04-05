import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game.game_loop import GameLoop
import pygame

class SmartCrossingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Initialize the game
        self.game = None
        self.render_mode = render_mode
        
        # Define action space (Discrete: 0=nothing, 1=left, 2=right, 3=accelerate, 4=brake)
        self.action_space = spaces.Discrete(5)
        
        # Define observation space
        # [x, y, angle, speed, goal_x, goal_y, nearest_obstacle_x, nearest_obstacle_y]
        low = np.array([0, 0, -360, -3, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1024, 768, 360, 150, 1024, 768, 1024, 768], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the game to initial state
        self.game = GameLoop()
        
        # Get initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
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
        
        # Update game state
        dt = 1.0 / 60.0  # Fixed timestep
        self.game.car.update(keys, dt)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = False
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """Convert game state to observation array"""
        car = self.game.car
        goal = self.game.goal
        obstacle = self.game.obstacle
        
        return np.array([
            car.x,
            car.y,
            car.angle,
            car.speed,
            goal.x,
            goal.y,
            obstacle.x,
            obstacle.y,
        ], dtype=np.float32)

    def _get_info(self):
        """Get additional information about the environment state"""
        car = self.game.car
        goal = self.game.goal
        
        # Calculate distance to goal
        distance_to_goal = np.sqrt(
            (car.x - goal.x) ** 2 + 
            (car.y - goal.y) ** 2
        )
        
        return {
            "distance_to_goal": distance_to_goal
        }

    def _calculate_reward(self):
        """Calculate the reward based on current state"""
        reward = 0
        car = self.game.car
        goal = self.game.goal
        obstacle = self.game.obstacle
        
        # Reward for reaching goal
        if car.rect.colliderect(goal.rect):
            reward += 100
        
        # Penalty for collision with obstacle
        if car.rect.colliderect(obstacle.rect):
            reward -= 50
        
        # Small penalty for time
        reward -= 0.1
        
        # Reward for getting closer to goal (based on previous distance)
        current_distance = np.sqrt((car.x - goal.x) ** 2 + (car.y - goal.y) ** 2)
        if hasattr(self, '_previous_distance'):
            reward += (self._previous_distance - current_distance) * 0.1
        self._previous_distance = current_distance
        
        return reward

    def _is_terminated(self):
        """Check if episode should end"""
        car = self.game.car
        goal = self.game.goal
        obstacle = self.game.obstacle
        
        # End if car reaches goal or hits obstacle
        return (car.rect.colliderect(goal.rect) or 
                car.rect.colliderect(obstacle.rect))

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # The game already handles rendering
            pass

    def close(self):
        """Clean up resources"""
        if self.game:
            pygame.quit()
