# CS175 Project: Treasure Hunt Agent

This project implements a reinforcement learning agent that learns to navigate a treasure hunt course in Minecraft using the Malmo platform. The agent's goal is to collect diamonds and reach the emerald while avoiding death.

## Project Structure

- `train.py`: Main training script that implements the reinforcement learning loop. It handles:
  - Mission initialization and execution
  - State processing and reward calculation
  - Agent training and model saving
  - Episode management

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
python train.py
```

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

## Model Checkpoints

Trained models and metrics are saved in:
- `models/`: Contains saved model checkpoints
- `metrics/`: Contains training metrics in JSON format

## Customization

You can modify various parameters in `train.py`:
- `EPISODES`: Number of training episodes
- `MAX_STEPS_PER_EPISODE`: Maximum steps per episode
- `SAVE_INTERVAL`: How often to save the model
- `TARGET_UPDATE`: How often to update the target network

The reward structure can be modified in the `REWARDS` dictionary in `train.py`.

## Notes

- The agent uses a DQN algorithm with experience replay and target networks
- The state space consists of the agent's visual input (640x480 RGB images)
- The action space consists of four basic movements: forward, backward, left, and right
- The episode ends when either:
  - The agent collects the emerald
  - The agent dies
  - The maximum number of steps is reached 