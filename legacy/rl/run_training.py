import os
from rl.training import train

def main():
    """Run training with visualization"""
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Training parameters
    params = {
        'num_episodes': 500,      # Start with fewer episodes for testing
        'eval_interval': 10       # Evaluate every 10 episodes
    }
    
    print("Starting training with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print("\nTraining progress will be logged to Weights & Biases.")
    print("View progress at: https://wandb.ai")
    print("\nPress Ctrl+C to stop training at any time.")
    
    # Start training
    train(**params)

if __name__ == "__main__":
    main()
