# Smart Crossing

A simple 2D simulation game built with Pygame where cars navigate a road crossing and avoid obstacles. The project also integrates a reinforcement learning (RL) agent to learn how to drive the car through the environment.

## Getting Started

### Prerequisites

- Python (recommended version 3.13+)
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Sipsipl/smartcrossing.git
cd smartcrossing
```

2. Install dependencies with Poetry:
```bash
poetry install
```

3. Run the game:
```bash
poetry run python main.py --mode play
```

The game will launch in a 1024x768 window. Use the arrow keys to control the car and navigate through the environment while avoiding obstacles.

4. Run RL training (headless, ~2.9 minutes for 500 episodes):
```bash
poetry run python main.py --mode train
```

## Development

The project uses Poetry for dependency management. To add new dependencies:

```bash
poetry add package-name
```

To update dependencies:
```bash
poetry update
```


## RL Component

The `rl/` module contains a fully implemented DQN (Deep Q-Network) agent:

- `agent.py`: DQN agent with experience replay and epsilon-greedy exploration
- `environment.py`: Gymnasium-compatible wrapper around the game
- `models.py`: 3-layer fully connected neural network (8 → 128 → 128 → 5)
- `training.py`: Training loop with W&B logging and model checkpointing

**Dependencies:** PyTorch, Gymnasium, Weights & Biases

## Training Process

The training process follows this hierarchical structure:

```
📦 Run (500 Episodes)
│
├── 🔁 Episode
│   ├─ 🔄 Step/Frame (60 FPS)
│   │    ├─ Choose Action (agent.select_action)
│   │    ├─ Update Environment (env.step)
│   │    ├─ Calculate Reward
│   │    ├─ Observe New State
│   │    ├─ Store Experience in Replay Buffer
│   │    └─ Perform Training (when Buffer > 64)
│   └─ 🏁 Episode End: Goal Reached or Collision
│
├── 📊 Evaluation (every 10 Episodes)
│   ├─ 5 Evaluation Episodes
│   └─ Without Exploration (ε = 0)
│
└── 💾 Model Storage
    ├─ Best Model (best_model.pt)
    └─ Final Model (final_model.pt)
```

Each training run consists of 500 episodes. During each episode, the agent interacts with the environment at 60 frames per second until it either reaches the goal or collides with an obstacle. Every 10 episodes, the agent is evaluated without exploration to measure its true performance. The best performing model is saved throughout training, and the final model is saved when training completes.

### Evaluation Process
Every 10 training episodes, the training process pauses to evaluate the agent's performance:
1. The agent runs 5 evaluation episodes
2. During evaluation:
   - The agent uses a fully greedy policy (ε = 0)
   - Each episode runs until goal reached or collision
   - The average reward across all 5 episodes is calculated
3. If the evaluation reward is higher than the previous best:
   - The current model is saved as 'best_model.pt'
   - The reward is logged to Weights & Biases
4. This process repeats 50 times during training (500/10 = 50 evaluation points)

The evaluation process helps:
- Measure true agent performance without random exploration
- Track learning progress over time
- Save the best performing model for later use

## Training Time Estimation

Training runs in headless mode by default (`render_mode=None`):

- Average episode duration: ~0.23 seconds
- Total training episodes: 500
- Estimated total training time: ~2.9 minutes

> Rendering during training (~9.2 hours for 500 episodes) can be enabled by passing `render_mode="human"` to `SmartCrossingEnv`, but is not recommended.

## Reward System

The AI agent's reward system works as follows:
- +100 points for reaching the goal
- -50 points for collision with obstacles
- -0.1 points per time unit (motivation for quick execution)
- Small rewards for approaching the goal

This structure motivates the agent to reach the goal quickly and safely.

## Observation Space

The agent receives the following information at each time step:
- Car position (X, Y)
- Car angle (-360° to 360°)
- Car velocity (-3 to 150)
- Goal position (X, Y)
- Obstacle position (X, Y)

These 8-dimensional values are provided as normalized values in the range [0, 1024] for X-coordinates and [0, 768] for Y-coordinates.
