# SmartCrossing Tests

This directory contains tests for the SmartCrossing project.

## Running Tests

To run all tests:

```bash
pytest
```

To run tests with coverage report:

```bash
pytest --cov=game --cov=rl --cov-report=term-missing
```

To run a specific test file:

```bash
pytest tests/test_game_loop.py
```

To run a specific test function:

```bash
pytest tests/test_game_loop.py::test_initialization
```

## Test Structure

- `test_game_loop.py`: Tests for the game loop and pygame initialization
- `test_environment.py`: Tests for the RL environment
- `test_agent.py`: Tests for the DQN agent and replay buffer
- `conftest.py`: Shared fixtures for all tests

## Test Fixtures

The `conftest.py` file contains shared fixtures:

- `cleanup_pygame`: Automatically cleans up pygame before and after each test
- `test_models_dir`: Creates and cleans up a directory for test model files 