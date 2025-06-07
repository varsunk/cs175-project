# CS175 Project: Treasure Hunt Agent

This project implements a reinforcement learning agent that learns to navigate a treasure hunt course in Minecraft using the Malmo platform. The agent's goal is to collect diamonds and reach the emerald while avoiding death.

## Project Structure

- `run.py`: Main training script that implements the reinforcement learning loop. It handles:
  - Mission initialization and execution
  - State processing and reward calculation
  - Agent training and model saving
  - Episode management
  - Multiple model training support
  - Checkpoint management

- `agent.py`: Contains the Agent class that implements the DQN (Deep Q-Network) algorithm. It handles:
  - Neural network architecture
  - Experience replay buffer
  - Action selection (epsilon-greedy)
  - Model training and saving
  - Target network updates

- `dqn.py`: Implements the neural network architecture for the DQN algorithm.

- `replay_buffer.py`: Implements the experience replay buffer for storing and sampling transitions.

- `envs/treasure_hunt.xml`: Malmo mission specification file that defines:
  - The Minecraft world to use
  - Item spawn locations (diamonds and emerald)
  - Agent starting position and inventory
  - Reward structure
  - Observation handlers

## Reward Structure

The agent receives the following rewards:
- +10 for collecting diamonds
- +50 for collecting emeralds (also ends episode)
- -200 for dying
- -1 per step as a time penalty
- -1 per command as a movement penalty

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Malmo Platform (version 0.37.0 or compatible)
- Minecraft Java Edition

## Setup

1. Install the Malmo platform following the instructions at https://github.com/microsoft/malmo
2. Install Python dependencies:
   ```bash
   pip install torch numpy
   ```
3. Make sure you have the correct Minecraft world save file and update the path in `envs/treasure_hunt.xml`

## Running the Training

To start training the agent:

```bash
python run.py
```

### Training Multiple Models

The script supports training multiple models with different names. Each model will have its own separate directories for checkpoints and metrics.

To train a new model with a specific name:
```bash
python run.py --model-name my_new_model
```

To train another model with a different name:
```bash
python run.py --model-name another_model
```

To load a specific model's checkpoint:
```bash
python run.py --model-name my_new_model --load-checkpoint latest
```

To start training from scratch with a new model name:
```bash
python run.py --model-name fresh_model --no-checkpoint
```

### Command Line Arguments

- `--model-name`: Name of the model to train (creates separate directories for checkpoints and metrics)
- `--load-checkpoint`: Load checkpoint: "latest", "best", or path to specific checkpoint file
- `--no-checkpoint`: Start training from scratch, ignoring any existing checkpoints

The training script will:
1. Initialize the DQN agent
2. Run episodes of the treasure hunt mission
3. Save the model and metrics every 100 episodes
4. Display training progress including:
   - Current episode number
   - Step count
   - Current reward
   - Total reward
   - Epsilon value (exploration rate)
   - Completion times and rankings

## Model Checkpoints

Trained models and metrics are saved in model-specific directories:
- `models/<model_name>/`: Contains saved model checkpoints for each model
- `metrics/<model_name>/`: Contains training metrics in JSON format for each model

Each model directory contains:
- `latest_checkpoint.pth`: Most recent model checkpoint
- `best_checkpoint.pth`: Best performing model checkpoint
- `latest_metrics.json`: Most recent training metrics
- `best_metrics.json`: Best performing model metrics
- `completion_times.txt`: Record of completion times and rankings

## Customization

You can modify various parameters in `run.py`:
- `EPISODES`: Number of training episodes
- `MAX_STEPS_PER_EPISODE`: Maximum steps per episode
- `SAVE_INTERVAL`: How often to save the model
- `TARGET_UPDATE`: How often to update the target network
- `CHECKPOINT_INTERVAL`: How often to save checkpoints
- `BEST_MODEL_THRESHOLD`: Minimum episodes before considering saving best model

The reward structure can be modified in the `REWARDS` dictionary in `run.py`.

## Notes

- The agent uses a DQN algorithm with experience replay and target networks
- The state space consists of the agent's visual input (640x480 RGB images)
- The action space consists of four basic movements: forward, backward, left, and right
- The episode ends when either:
  - The agent collects the emerald
  - The agent dies
  - The maximum number of steps is reached
- Each model maintains its own:
  - Training history
  - Best completion times
  - Checkpoints
  - Metrics 