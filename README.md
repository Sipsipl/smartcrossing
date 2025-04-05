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
poetry run python main.py
```

The game will launch in a 1024x768 window. Use the arrow keys to control the car and navigate through the environment while avoiding obstacles.

## Development

The project uses Poetry for dependency management. To add new dependencies:

```bash
poetry add package-name
```

To update dependencies:
```bash
poetry update
```


## Plan to implement RL component

The reinforcement learning component will be implemented in the following stages:

1. **Project Structure**
   - Create new `rl` module with:
     - `agent.py`: Implementation of the RL agent
     - `environment.py`: Game environment wrapper for RL
     - `models.py`: Neural network models
     - `training.py`: Training loop and utilities

2. **Environment Implementation**
   - Wrap existing game into a Gym-like environment
   - Define:
     - State space: Car position, velocity, distance to goal, obstacles
     - Action space: Discrete actions (accelerate, brake, steer left/right)
     - Reward function: Based on goal reaching, collision avoidance, time
     - Reset and step functions

3. **Agent Implementation**
   - Implement DQN (Deep Q-Network) agent
   - Key components:
     - Experience replay buffer
     - Target network for stable learning
     - Epsilon-greedy exploration

4. **Game Loop Integration**
   - Support both manual and agent control modes
   - Add visualization of agent's decision-making

5. **Training Pipeline**
   - Episode management
   - Model checkpointing
   - Performance metrics logging
   - Multiple training scenarios

6. **Dependencies**
   - PyTorch: Neural network implementation
   - Gymnasium: Environment structure
   - Weights and Biases: Training visualization
