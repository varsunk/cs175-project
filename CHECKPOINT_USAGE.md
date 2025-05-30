# Checkpoint Loading Feature

The training script now supports loading from checkpoints, allowing you to resume training from where you left off instead of starting from scratch.

## Quick Start

### Resume from last checkpoint (automatic)
```bash
python run.py
```
This will automatically look for and load the latest checkpoint if available.

### Start fresh training (ignore checkpoints)
```bash
python run.py --no-checkpoint
```

### Load best checkpoint
```bash
python run.py --load-checkpoint best
```

### Load specific checkpoint file
```bash
python run.py --load-checkpoint /path/to/your/checkpoint.pth
```

## Checkpoint Types

The system saves two types of checkpoints:

1. **Latest Checkpoint** (`latest_checkpoint.pth`)
   - Saved every 50 episodes
   - Contains the most recent training state
   - Use this to resume interrupted training

2. **Best Checkpoint** (`best_checkpoint.pth`)
   - Saved whenever a new best reward is achieved
   - Contains the model state from the best performing episode
   - Use this to continue training from your best model

## What Gets Saved

Each checkpoint includes:
- Episode number
- Model weights (policy and target networks)
- Optimizer state
- Current epsilon value
- Current and best rewards
- Training metrics and completion times

## Checking Available Checkpoints

Run the checkpoint scanner to see what's available:
```bash
python checkpoint_demo.py
```

This will show:
- Available checkpoints and their details
- Training progress summary
- Completion statistics
- Usage examples

## File Structure

```
models/
├── latest_checkpoint.pth     # Most recent checkpoint
├── best_checkpoint.pth       # Best performing checkpoint
└── dqn_model_episode_*.pth   # Episode-specific saves

metrics/
├── latest_metrics.json       # Metrics for latest checkpoint
├── best_metrics.json         # Metrics for best checkpoint
└── completion_times.txt      # Human-readable completion log
```

## Command Line Arguments

The script accepts these arguments for checkpoint management:

- `--load-checkpoint TYPE_OR_PATH`: Load specific checkpoint
  - `latest`: Load latest checkpoint
  - `best`: Load best checkpoint
  - `/path/to/file.pth`: Load custom checkpoint file
- `--no-checkpoint`: Start training from scratch, ignore existing checkpoints

## Examples

### Resume training that was interrupted
```bash
# This will automatically load the latest checkpoint
python run.py
```

### Continue training from your best model
```bash
python run.py --load-checkpoint best
```

### Start completely fresh (useful for testing different hyperparameters)
```bash
python run.py --no-checkpoint
```

### Load a specific checkpoint file
```bash
python run.py --load-checkpoint models/latest_checkpoint.pth
```

## Benefits

1. **Resume interrupted training**: No need to start over if training crashes or is stopped
2. **Experiment with different approaches**: Start from best checkpoint and try different hyperparameters
3. **Save training time**: Continue from good models instead of retraining from scratch
4. **Preserve progress**: All metrics and completion times are maintained across sessions

## Tips

- The system automatically loads the latest checkpoint by default if available
- Use `--no-checkpoint` when you want to test completely new hyperparameters
- Check `checkpoint_demo.py` output to see training progress before resuming
- Best checkpoints are only saved when performance actually improves
- All Malmo command line arguments still work (they're passed through automatically)

## Troubleshooting

If checkpoint loading fails:
- The script will automatically fall back to training from scratch
- Check that the checkpoint files aren't corrupted
- Ensure you have the same Python/PyTorch environment
- Use `--no-checkpoint` to bypass any checkpoint issues 