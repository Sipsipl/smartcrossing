import argparse
import os
from rl.training import train

def main():
    """Run training"""
    parser = argparse.ArgumentParser(description="Train the SmartCrossing RL agent")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--eval-interval", type=int, default=10, help="Episodes between evaluations")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during training (slow).",
    )
    args = parser.parse_args()

    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Training parameters
    params = {
        "num_episodes": args.episodes,
        "eval_interval": args.eval_interval,
        "render_mode": "human" if args.render else None,
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
