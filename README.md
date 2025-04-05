# Smart Crossing

A simple 2D simulation game built with Pygame where cars navigate a road crossing and avoid obstacles. The project also integrates a reinforcement learning (RL) agent to learn how to drive the car through the environment.

## Getting Started

### Prerequisites

- Python (recommended version 3.8+)
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
