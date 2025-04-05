import os
from datetime import datetime
import numpy as np
import torch
import wandb
from .environment import SmartCrossingEnv
from .agent import DQNAgent

def train(num_episodes: int = 1000, eval_interval: int = 10):
    """Train the DQN agent
    
    Args:
        num_episodes: Number of episodes to train for
        eval_interval: Number of episodes between evaluations
    """
    # Initialize wandb
    wandb.init(
        project="smartcrossing",
        config={
            "num_episodes": num_episodes,
            "eval_interval": eval_interval,
            "architecture": "DQN",
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "batch_size": 64,
        }
    )
    
    # Create environment and agent
    env = SmartCrossingEnv(render_mode="human")  # Enable rendering
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
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
            
            # Render the environment
            env.render()
        
        # Log training metrics
        episode_rewards.append(episode_reward)
        metrics = {
            "train/episode_reward": episode_reward,
            "train/epsilon": agent.epsilon,
        }
        if loss is not None:
            metrics["train/loss"] = loss
        wandb.log(metrics)
        
        # Evaluate and save best model
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate(env, agent)
            wandb.log({"eval/episode_reward": eval_reward})
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save('models/best_model.pt')
                wandb.save('models/best_model.pt')
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Average Training Reward: {np.mean(episode_rewards[-eval_interval:]):.2f}")
            print(f"Evaluation Reward: {eval_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print("----------------------------------------")
    
    # Final save
    agent.save('models/final_model.pt')
    wandb.save('models/final_model.pt')
    wandb.finish()
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
            env.render()  # Enable rendering during evaluation too
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Start training
    train()
