import numpy as np
import torch
import os
from rl.environment import SmartCrossingEnv
from rl.agent import DQNAgent, ReplayBuffer
from rl.models import DQN

def test_dqn_model():
    """Test DQN model initialization and forward pass"""
    # Create model
    input_dim = 8  # State dimension
    output_dim = 5  # Action dimension
    model = DQN(input_dim, output_dim)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    
    # Verify output shape
    assert output.shape == (batch_size, output_dim), f"Expected shape {(batch_size, output_dim)}, got {output.shape}"

def test_replay_buffer():
    """Test replay buffer functionality"""
    # Create buffer
    buffer = ReplayBuffer(capacity=100)
    
    # Add experiences
    for i in range(10):
        state = np.random.random(8)
        action = np.random.randint(5)
        reward = np.random.random()
        next_state = np.random.random(8)
        done = bool(np.random.randint(2))
        buffer.push(state, action, reward, next_state, done)
    
    # Test sampling
    batch_size = 4
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    # Verify shapes
    assert states.shape == (batch_size, 8), f"Expected state shape (4, 8), got {states.shape}"
    assert actions.shape == (batch_size,), f"Expected action shape (4,), got {actions.shape}"
    assert rewards.shape == (batch_size,), f"Expected reward shape (4,), got {rewards.shape}"
    assert next_states.shape == (batch_size, 8), f"Expected next_state shape (4, 8), got {next_states.shape}"
    assert dones.shape == (batch_size,), f"Expected done shape (4,), got {dones.shape}"

def test_agent():
    """Test DQN agent functionality"""
    # Create environment and agent
    env = SmartCrossingEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    # Test action selection
    state = env.reset()[0]
    action = agent.select_action(state)
    assert 0 <= action < action_dim, f"Action {action} out of bounds [0, {action_dim})"
    
    # Test experience collection and training
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    agent.memory.push(state, action, reward, next_state, done)
    
    # Add more experiences for training
    for _ in range(63):  # Need batch_size experiences for training
        state = np.random.random(state_dim)
        action = np.random.randint(action_dim)
        reward = np.random.random()
        next_state = np.random.random(state_dim)
        done = bool(np.random.randint(2))
        agent.memory.push(state, action, reward, next_state, done)
    
    # Test training step
    loss = agent.train_step()
    assert loss is not None, "Training step failed to return loss"

def test_save_load(test_models_dir):
    """Test model saving and loading"""
    # Create agent and save
    env = SmartCrossingEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    model_path = os.path.join(test_models_dir, 'test_model.pt')
    agent.save(model_path)
    
    # Create new agent and load
    new_agent = DQNAgent(state_dim, action_dim)
    new_agent.load(model_path)
    
    # Verify weights are the same
    for p1, p2 in zip(agent.policy_net.parameters(), new_agent.policy_net.parameters()):
        assert torch.equal(p1, p2), "Model parameters don't match after loading"
