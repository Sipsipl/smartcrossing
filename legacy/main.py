import argparse
from game.game_loop import GameLoop
from rl.run_training import main as train_rl

def main():
    parser = argparse.ArgumentParser(description='SmartCrossing - Traffic Management System')
    parser.add_argument('--mode', choices=['play', 'train'], default='play',
                       help='Run mode: play the game or train the RL agent')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_rl()
    else:
        game = GameLoop()
        game.run()

if __name__ == "__main__":
    main()
