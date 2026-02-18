# CLAUDE.md

## Commands

```bash
# Install dependencies (required before anything else)
poetry install

# Run tests
poetry run pytest tests/ -v

# Run the game
poetry run python main.py --mode play

# Run RL training (headless, fast)
poetry run python main.py --mode train
```

## Notes

- Python 3.13 required (`pyproject.toml` enforces `^3.13.2`); poetry picks it up automatically
- Training runs headless (`render_mode=None`) for speed — no display needed
- Models are saved to `models/best_model.pt` and `models/final_model.pt`
- Training logs to Weights & Biases (`wandb`) — a W&B account is needed or set `WANDB_MODE=offline`
