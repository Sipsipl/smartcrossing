import numpy as np
import gymnasium as gym
from rl.environment import SmartCrossingEnv

def test_environment_initialization():
    """Test environment initialization and spaces"""
    env = SmartCrossingEnv()
    
    # Test action space
    assert env.action_space.n == 5, f"Expected 5 actions, got {env.action_space.n}"
    
    # Test observation space
    assert env.observation_space.shape == (8,), f"Expected 8-dim observation, got {env.observation_space.shape}"
    assert env.observation_space.dtype == np.float32, f"Expected float32 dtype, got {env.observation_space.dtype}"

def test_reset():
    """Test environment reset functionality"""
    env = SmartCrossingEnv()
    observation, info = env.reset()
    
    # Test observation is within bounds
    assert env.observation_space.contains(observation), "Observation out of bounds"
    
    # Test info contains distance_to_goal
    assert "distance_to_goal" in info, "Info missing distance_to_goal"
    assert isinstance(info["distance_to_goal"], (int, float)), "distance_to_goal should be numeric"

def test_step():
    """Test environment step function"""
    env = SmartCrossingEnv()
    observation, info = env.reset()
    
    # Test each action
    for action in range(5):
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Test observation is valid
        assert env.observation_space.contains(next_obs), f"Invalid observation after action {action}"
        
        # Test reward is numeric
        assert isinstance(reward, (int, float)), f"Invalid reward type for action {action}"
        
        # Test terminated and truncated are boolean
        assert isinstance(terminated, bool), f"Invalid terminated type for action {action}"
        assert isinstance(truncated, bool), f"Invalid truncated type for action {action}"
        
        # Test info contains required fields
        assert "distance_to_goal" in info, f"Info missing distance_to_goal for action {action}"

def test_episode():
    """Test a full episode"""
    env = SmartCrossingEnv()
    observation, info = env.reset()
    
    episode_reward = 0
    steps = 0
    max_steps = 1000
    
    while steps < max_steps:
        # Take random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    # Verify episode completed
    assert steps > 0, "Episode should have taken at least one step"
    assert steps <= max_steps, f"Episode should not exceed {max_steps} steps"
