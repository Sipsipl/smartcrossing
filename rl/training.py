import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .environment import SmartCrossingEnv
from .agent import DQNAgent

def train(num_episodes: int = 1000, eval_interval: int = 10):
    """Train the DQN agent
    
    Args:
        num_episodes: Number of episodes to train for
        eval_interval: Number of episodes between evaluations
    """
    # Create environment and agent
    env = SmartCrossingEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    # Set up logging
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'DQN_{current_time}')
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_eval_reward = float('-inf')
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition and train
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Log training metrics
        episode_rewards.append(episode_reward)
        writer.add_scalar('Training/Episode Reward', episode_reward, episode)
        if loss is not None:
            writer.add_scalar('Training/Loss', loss, episode)
        
        # Evaluate and save best model
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate(env, agent)
            writer.add_scalar('Evaluation/Episode Reward', eval_reward, episode)
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save('models/best_model.pt')
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Average Training Reward: {np.mean(episode_rewards[-eval_interval:]):.2f}")
            print(f"Evaluation Reward: {eval_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print("----------------------------------------")
    
    # Final save
    agent.save('models/final_model.pt')
    writer.close()
    env.close()

def evaluate(env: SmartCrossingEnv, agent: DQNAgent, num_episodes: int = 5) -> float:
    """Evaluate the agent's performance
    
    Args:
        env: Environment to evaluate in
        agent: Agent to evaluate
        num_episodes: Number of episodes to evaluate for
        
    Returns:
        Average reward over evaluation episodes
    """
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Start training
    train()
