import os
from datetime import datetime
import numpy as np
import torch
import wandb
import traceback
from .environment import SmartCrossingEnv
from .agent import DQNAgent
from .logger import setup_logger

# Set up logger
logger = setup_logger()

def train(num_episodes: int = 1000, eval_interval: int = 10):
    """Train the DQN agent
    
    Args:
        num_episodes: Number of episodes to train for
        eval_interval: Number of episodes between evaluations
    """
    try:
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
        
        logger.info("Creating environment and agent...")
        # Create environment and agent
        env = SmartCrossingEnv(render_mode="human")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = DQNAgent(state_dim, action_dim)
        
        # Training loop
        best_eval_reward = float('-inf')
        episode_rewards = []
        
        logger.info("Starting training loop...")
        for episode in range(num_episodes):
            try:
                state, _ = env.reset()
                episode_reward = 0
                done = False
                step = 0
                
                while not done:
                    try:
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
                        
                        step += 1
                        if step % 100 == 0:  # Log every 100 steps
                            logger.debug(f"Episode {episode + 1}, Step {step}: Reward={reward:.2f}, Loss={loss if loss else 'None'}")
                    
                    except Exception as e:
                        logger.error(f"Error during training step: {str(e)}")
                        logger.debug(traceback.format_exc())
                        raise
                
                # Log training metrics
                episode_rewards.append(episode_reward)
                metrics = {
                    "train/episode_reward": episode_reward,
                    "train/epsilon": agent.epsilon,
                }
                if loss is not None:
                    metrics["train/loss"] = loss
                wandb.log(metrics)
                
                logger.info(f"Episode {episode + 1} completed: Reward={episode_reward:.2f}, Epsilon={agent.epsilon:.3f}")
                
                # Evaluate and save best model
                if (episode + 1) % eval_interval == 0:
                    try:
                        eval_reward = evaluate(env, agent)
                        wandb.log({"eval/episode_reward": eval_reward})
                        
                        if eval_reward > best_eval_reward:
                            best_eval_reward = eval_reward
                            agent.save('models/best_model.pt')
                            wandb.save('models/best_model.pt')
                            logger.info(f"New best model saved with reward: {eval_reward:.2f}")
                        
                        logger.info(f"Evaluation completed:")
                        logger.info(f"  Average Training Reward: {np.mean(episode_rewards[-eval_interval:]):.2f}")
                        logger.info(f"  Evaluation Reward: {eval_reward:.2f}")
                        logger.info(f"  Epsilon: {agent.epsilon:.3f}")
                        logger.info("----------------------------------------")
                    
                    except Exception as e:
                        logger.error(f"Error during evaluation: {str(e)}")
                        logger.debug(traceback.format_exc())
                        raise
            
            except Exception as e:
                logger.error(f"Error during episode {episode + 1}: {str(e)}")
                logger.debug(traceback.format_exc())
                raise
        
        # Final save
        agent.save('models/final_model.pt')
        wandb.save('models/final_model.pt')
        logger.info("Training completed successfully")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.debug(traceback.format_exc())
        raise
    
    finally:
        wandb.finish()
        env.close()
        logger.info("Environment closed")

def evaluate(env: SmartCrossingEnv, agent: DQNAgent, num_episodes: int = 5) -> float:
    """Evaluate the agent's performance
    
    Args:
        env: Environment to evaluate in
        agent: Agent to evaluate
        num_episodes: Number of episodes to evaluate for
        
    Returns:
        Average reward over evaluation episodes
    """
    logger.info(f"Starting evaluation over {num_episodes} episodes...")
    rewards = []
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                env.render()
                
                step += 1
                if step % 100 == 0:  # Log every 100 steps
                    logger.debug(f"Eval Episode {episode + 1}, Step {step}: Reward={reward:.2f}")
            
            rewards.append(episode_reward)
            logger.info(f"Eval Episode {episode + 1} completed: Reward={episode_reward:.2f}")
        
        avg_reward = np.mean(rewards)
        logger.info(f"Evaluation completed with average reward: {avg_reward:.2f}")
        return avg_reward
    
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Start training
    train()
