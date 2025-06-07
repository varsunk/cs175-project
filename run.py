import MalmoPython # type: ignore
import time
import json
import numpy as np
from agent import Agent
import os
from pathlib import Path
import sys
import torch
import gc
import heapq  # For maintaining top 5 fastest times
import argparse  # For command line arguments

# Initialize CUDA if available
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
    # Set default tensor type to CUDA
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA is not available, using CPU")
    torch.set_default_tensor_type('torch.FloatTensor')

# Constants
XML_FILE = "./envs/treasure_hunt.xml"
EPISODES = 5000
MAX_STEPS_PER_EPISODE = 5000
# Reduce input size significantly to avoid memory issues
INPUT_SHAPE = (120, 160, 3)  # Reduced from (480, 640, 3) to save memory
SAVE_INTERVAL = 10
TARGET_UPDATE = 10

# Add new constants for checkpoint saving
CHECKPOINT_INTERVAL = 50  # Save latest checkpoint every 50 episodes
BEST_MODEL_THRESHOLD = 10  # Only consider saving best model after this many episodes

# Define actions (turning and jumping to navigate the race track)
ACTIONS = [
    "turn_right",      # Turn right
    "turn_left",       # Turn left
    "no_turn",         # Continue straight (no turning)
    "jump",            # Jump while running (useful on ice blocks)
]

# Action mapping to continuous movement commands
ACTION_COMMANDS = {
    "turn_right": ("turn", 0.5),
    "turn_left": ("turn", -0.5),
    "no_turn": ("turn", 0),
    "jump": ("jump", 1),
}

# Reward constants
REWARDS = {
    "time_penalty": -1,     # Very small time penalty to encourage efficiency
    "movement_bonus": 1,     # Small bonus for forward movement
}

def create_mission_spec():
    xml = Path(XML_FILE).read_text()
    mission_spec = MalmoPython.MissionSpec(xml, True)
    mission_spec.allowAllContinuousMovementCommands()
    return mission_spec

def process_frame(frame):
    pixels = np.array(frame.pixels, dtype=np.uint8)
    frame_shape = (frame.height, frame.width, frame.channels)
    image = pixels.reshape(frame_shape)
    
    # Resize image to reduce memory usage - use simple downsampling
    # From original size to target INPUT_SHAPE
    target_height, target_width = INPUT_SHAPE[0], INPUT_SHAPE[1]
    
    # Simple downsampling by taking every nth pixel
    height_step = max(1, frame.height // target_height)
    width_step = max(1, frame.width // target_width)
    
    # Downsample the image
    resized = image[::height_step, ::width_step, :]
    
    # Ensure we get exactly the target size by cropping if necessary
    resized = resized[:target_height, :target_width, :]
    
    # If the downsampled image is smaller than target, pad it
    if resized.shape[0] < target_height or resized.shape[1] < target_width:
        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1], :] = resized
        resized = padded
    
    # Normalize pixel values to [0, 1]
    resized = resized.astype(np.float32) / 255.0
    
    # Transpose from (H, W, C) to (C, H, W) for PyTorch
    image = np.transpose(resized, (2, 0, 1))
    return image

def process_reward(world_state, step, prev_z_pos):
    reward = 0
    episode_end = False
    reached_finish = False
    current_z_pos = prev_z_pos  # Keep for compatibility
    
    # Get rewards from Malmo's reward system
    for reward_obj in world_state.rewards:
        malmo_reward = reward_obj.getValue()
        reward += malmo_reward
        print(f"Malmo reward received: {malmo_reward}")
        
        # Check if this is the emerald block reward (finish line)
        if malmo_reward > 5000:  # Emerald block reward from XML
            reached_finish = True
            print("🏁 FINISH LINE REACHED! Agent touched emerald block!")
    
    if len(world_state.observations):
        obs = json.loads(world_state.observations[-1].text)
        
        
        # Debug position every 20 steps
        if "XPos" in obs and "ZPos" in obs and step % 20 == 0:
            print(f"Agent position: X={obs['XPos']:.1f}, Y={obs.get('YPos', 0):.1f}, Z={obs['ZPos']:.1f}")
            
        # Movement detection for debugging
        if "XPos" in obs and "ZPos" in obs:
            current_x, current_z = obs["XPos"], obs["ZPos"]
            y_pos = obs.get("YPos", 0)
            
            if hasattr(process_reward, 'last_pos'):
                last_x, last_z = process_reward.last_pos
                distance_moved = ((current_x - last_x)**2 + (current_z - last_z)**2)**0.5
                if step % 20 == 0:  # Only print movement info every 20 steps
                    if distance_moved < 0.1:
                        print(f"Warning: Agent seems stuck at X={current_x:.1f}, Z={current_z:.1f}, Y={y_pos:.1f}")
                    else:
                        print(f"Movement detected: moved {distance_moved:.2f} blocks")
            process_reward.last_pos = (current_x, current_z)
            
        # Update z position for compatibility
        if "ZPos" in obs:
            current_z_pos = obs["ZPos"]
    else:
        print(f"Warning: No observations available at step {step}")
    
    # Add small time penalty to encourage efficiency
    reward += REWARDS["time_penalty"]
    
    return reward, episode_end, current_z_pos, reached_finish

def get_action_reward(action):
    """Give bonus reward for forward movement and penalize excessive turning"""
    action_name = ACTIONS[action]
    if action_name == "no_turn":  # Going straight - encourage this
        return REWARDS["movement_bonus"]
    elif action_name == "jump":  # Jumping - bonus for speed on ice blocks
        return REWARDS["movement_bonus"] * 2  # Double bonus for jumping
    elif action_name in ["turn_right", "turn_left"]:  # Turning - small penalty to discourage unnecessary turns
        return -5
    else:
        return 0.0

def save_checkpoint(agent, episode, reward, best_reward, best_completion_time, checkpoint_type="latest"):
    """Save checkpoint with proper naming"""
    try:
        if checkpoint_type == "latest":
            # Save latest checkpoint
            model_path = os.path.join(agent.model_dir, "latest_checkpoint.pth")
            metrics_path = os.path.join(agent.metrics_dir, "latest_metrics.json")
            
            torch.save({
                'episode': episode,
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'reward': reward,
                'best_reward': best_reward,
                'best_completion_time': best_completion_time
            }, model_path)
            
            # Save metrics
            with open(metrics_path, 'w') as f:
                json.dump(agent.metrics, f)
            
            print(f"Latest checkpoint saved at episode {episode} with reward {reward:.2f}")
            
        elif checkpoint_type == "best":
            # Save best checkpoint
            model_path = os.path.join(agent.model_dir, "best_checkpoint.pth")
            metrics_path = os.path.join(agent.metrics_dir, "best_metrics.json")
            
            torch.save({
                'episode': episode,
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'reward': reward,
                'best_reward': best_reward,
                'best_completion_time': best_completion_time
            }, model_path)
            
            # Save metrics
            with open(metrics_path, 'w') as f:
                json.dump(agent.metrics, f)
            
            print(f"🏆 NEW BEST MODEL saved at episode {episode} with reward {reward:.2f}!")
            
    except Exception as e:
        print(f"Error saving {checkpoint_type} checkpoint: {e}")

def load_checkpoint(agent, checkpoint_type="latest"):
    """Load checkpoint and return starting episode number, best reward, and best completion time"""
    try:
        if checkpoint_type == "latest":
            model_path = os.path.join(agent.model_dir, "latest_checkpoint.pth")
            metrics_path = os.path.join(agent.metrics_dir, "latest_metrics.json")
        elif checkpoint_type == "best":
            model_path = os.path.join(agent.model_dir, "best_checkpoint.pth")
            metrics_path = os.path.join(agent.metrics_dir, "best_metrics.json")
        else:
            # Custom checkpoint path
            model_path = checkpoint_type
            metrics_path = None
        
        if not os.path.exists(model_path):
            print(f"No checkpoint found at {model_path}")
            return 0, float('-inf'), [], float('inf')
        
        print(f"Loading checkpoint from {model_path}...")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model states
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        
        starting_episode = checkpoint['episode']
        best_reward = checkpoint['best_reward']
        current_reward = checkpoint['reward']
        
        # Load best completion time (with backward compatibility)
        loaded_best_completion_time = checkpoint.get('best_completion_time', float('inf'))
        
        # Load metrics if available
        top_times = []
        if metrics_path and os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                loaded_metrics = json.load(f)
                agent.metrics = loaded_metrics
                # Extract top times if they exist
                if 'top_5_times' in loaded_metrics:
                    # Convert back to heap format for top_completion_times
                    for time_record in loaded_metrics['top_5_times']:
                        top_times.append((-time_record['time'], time_record))
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"📊 Resuming from episode {starting_episode}")
        print(f"🎯 Current epsilon: {agent.epsilon:.3f}")
        print(f"🏆 Best reward so far: {best_reward:.2f}")
        print(f"📈 Last episode reward: {current_reward:.2f}")
        
        if agent.metrics.get('completion_times'):
            total_completions = len(agent.metrics['completion_times'])
            print(f"🏁 Total completions so far: {total_completions}")
            # Update best completion time from existing data if not in checkpoint
            if agent.metrics['completion_times'] and loaded_best_completion_time == float('inf'):
                loaded_best_completion_time = min(record['time'] for record in agent.metrics['completion_times'])
                print(f"🏁 Calculated best completion time from metrics: {loaded_best_completion_time:.2f} seconds")
            elif loaded_best_completion_time != float('inf'):
                print(f"🏁 Loaded best completion time: {loaded_best_completion_time:.2f} seconds")
        
        return starting_episode, best_reward, top_times, loaded_best_completion_time
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("Starting training from scratch...")
        return 0, float('-inf'), [], float('inf')

def update_top_times(top_times, episode, completion_time, reward):
    """Update the top 5 fastest completion times"""
    # Create a record for this completion
    time_record = {
        'episode': episode,
        'time': completion_time,
        'reward': reward,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add to heap (using negative time for min-heap to act as max-heap for times)
    heapq.heappush(top_times, (-completion_time, time_record))
    
    # Keep only top 5 fastest (smallest times)
    if len(top_times) > 5:
        heapq.heappop(top_times)  # Remove slowest time
    
    # Sort and return the current top 5 (fastest to slowest)
    sorted_times = sorted([record for _, record in top_times], key=lambda x: x['time'])
    
    return sorted_times

def print_top_times(top_times):
    """Print the current top 5 fastest times in a nice format"""
    if not top_times:
        return
    
    print("\n" + "="*60)
    print("🏁 TOP 5 FASTEST COMPLETION TIMES 🏁")
    print("="*60)
    
    sorted_times = sorted([record for _, record in top_times], key=lambda x: x['time'])
    
    for i, record in enumerate(sorted_times, 1):
        minutes = int(record['time'] // 60)
        seconds = record['time'] % 60
        print(f"{i}. Episode {record['episode']:4d} | "
              f"Time: {minutes:2d}m {seconds:05.2f}s | "
              f"Reward: {record['reward']:7.1f} | "
              f"{record['timestamp']}")
    
    print("="*60)

def save_completion_times_to_file(top_times, total_completions, total_episodes, completion_time=None, episode=None, reward=None, metrics_dir="metrics"):
    """Save completion times to a text file"""
    try:
        # Create metrics directory if it doesn't exist
        os.makedirs(metrics_dir, exist_ok=True)
        
        file_path = os.path.join(metrics_dir, "completion_times.txt")
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write("MINECRAFT MALMO TRAINING - COMPLETION TIMES LOG\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Episodes: {total_episodes}\n")
            f.write(f"Completed Episodes: {total_completions}\n")
            f.write(f"Completion Rate: {(total_completions/total_episodes)*100:.1f}%\n")
            f.write("=" * 60 + "\n\n")
            
            if completion_time and episode and reward:
                f.write("LATEST COMPLETION:\n")
                f.write("-" * 20 + "\n")
                minutes = int(completion_time // 60)
                seconds = completion_time % 60
                f.write(f"Episode {episode} completed in {minutes}m {seconds:.2f}s with reward {reward:.1f}\n\n")
            
            if top_times:
                f.write("TOP 5 FASTEST COMPLETION TIMES:\n")
                f.write("-" * 35 + "\n")
                
                sorted_times = sorted([record for _, record in top_times], key=lambda x: x['time'])
                
                for i, record in enumerate(sorted_times, 1):
                    minutes = int(record['time'] // 60)
                    seconds = record['time'] % 60
                    f.write(f"{i}. Episode {record['episode']:4d} | "
                            f"Time: {minutes:2d}m {seconds:05.2f}s | "
                            f"Reward: {record['reward']:7.1f} | "
                            f"{record['timestamp']}\n")
                
                f.write("-" * 35 + "\n")
                
                # Calculate average time of top 5
                if sorted_times:
                    avg_time = sum(record['time'] for record in sorted_times) / len(sorted_times)
                    avg_minutes = int(avg_time // 60)
                    avg_seconds = avg_time % 60
                    f.write(f"Average time of fastest runs: {avg_minutes}m {avg_seconds:.2f}s\n")
                    
                    fastest_time = sorted_times[0]['time']
                    fastest_minutes = int(fastest_time // 60)
                    fastest_seconds = fastest_time % 60
                    f.write(f"Personal best time: {fastest_minutes}m {fastest_seconds:.2f}s (Episode {sorted_times[0]['episode']})\n")
            else:
                f.write("No completed runs yet.\n")
        
        return file_path
        
    except Exception as e:
        print(f"Error saving completion times to file: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Malmo DQN Training with Checkpoint Support')
    parser.add_argument('--load-checkpoint', type=str, default=None, 
                       help='Load checkpoint: "latest", "best", or path to specific checkpoint file')
    parser.add_argument('--no-checkpoint', action='store_true', 
                       help='Start training from scratch, ignoring any existing checkpoints')
    parser.add_argument('--model-name', type=str, default='default',
                       help='Name of the model to train (creates separate directories for checkpoints and metrics)')
    
    # Parse known args to handle Malmo arguments
    args, unknown = parser.parse_known_args()
    
    print(f"Training model: {args.model_name}")
    
    # Create model-specific directories
    model_dir = os.path.join('models', args.model_name)
    metrics_dir = os.path.join('metrics', args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    print("Initializing agent...")
    agent = Agent(
        input_shape=INPUT_SHAPE,
        n_actions=len(ACTIONS),
        alpha=0.001,           # Slightly higher learning rate
        gamma=0.90,            # Higher discount factor for longer-term planning
        epsilon=0.7,           # Start with lower epsilon (less random)
        epsilon_decay=0.995,   # Slower decay
        epsilon_min=0.1,       # Higher minimum epsilon to maintain some exploration
        batch_size=16,         # Reduced from 32 to 16 to save memory
        mem_size=2000,         # Reduced from 5000 to 2000 to save memory
        model_dir=model_dir,   # Pass model-specific directory
        metrics_dir=metrics_dir  # Pass metrics-specific directory
    )
    
    # Initialize checkpoint tracking
    starting_episode = 0
    best_reward = float('-inf')
    top_completion_times = []
    total_completions = 0
    best_completion_time = float('inf')  # Track the best completion time for bonus rewards
    
    # Handle checkpoint loading
    if not args.no_checkpoint:
        if args.load_checkpoint:
            # Load specific checkpoint
            starting_episode, best_reward, top_completion_times, best_completion_time = load_checkpoint(agent, args.load_checkpoint)
        else:
            # Try to load latest checkpoint automatically
            print("Checking for existing checkpoints...")
            starting_episode, best_reward, top_completion_times, best_completion_time = load_checkpoint(agent, "latest")
        
        # Update total completions from loaded metrics
        if agent.metrics.get('completion_times'):
            total_completions = len(agent.metrics['completion_times'])
    else:
        print("Starting training from scratch (--no-checkpoint specified)")
    
    print("Initializing Malmo...")
    agent_host = MalmoPython.AgentHost()
    try:
        # Pass unknown args to Malmo (this preserves Malmo's argument parsing)
        agent_host.parse(sys.argv[:1] + unknown)  # Keep script name + unknown Malmo args
    except RuntimeError as e:
        print("ERROR:", e)
        print(agent_host.getUsage())
        exit(1)
    
    agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
    agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
    
    # Adjust episode range if resuming from checkpoint
    episode_range = range(starting_episode, EPISODES)
    if starting_episode > 0:
        print(f"🔄 Resuming training from episode {starting_episode + 1} to {EPISODES}")
        print(f"📋 Episodes remaining: {EPISODES - starting_episode}")
    else:
        print(f"🆕 Starting fresh training for {EPISODES} episodes")
    
    for episode in episode_range:
        print(f"\nEpisode {episode + 1}/{EPISODES}")
        
        mission = create_mission_spec()
        record = MalmoPython.MissionRecordSpec()
        
        max_retries = 3
        for retry in range(max_retries):
            try:
                agent_host.startMission(mission, record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                time.sleep(2)
        
        print("Waiting for mission to start...")
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
        
        print("Mission started!")
        
        # Start timing the episode
        episode_start_time = time.time()
        reached_finish_line = False
        
        # Check starting position
        world_state = agent_host.getWorldState()
        if len(world_state.observations):
            obs = json.loads(world_state.observations[-1].text)
            start_x = obs.get("XPos", "unknown")
            start_y = obs.get("YPos", "unknown") 
            start_z = obs.get("ZPos", "unknown")
            print(f"Agent starting position: X={start_x}, Y={start_y}, Z={start_z}")
        else:
            print("Warning: No initial observations available")
        
        # Start continuous forward movement
        print("Starting continuous forward movement...")
        agent_host.sendCommand("move 1")
        
        # Give time for mission to fully initialize
        time.sleep(0.5)
        
        total_reward = 0
        step = 0
        previous_z_pos = None  # Initialize position tracking for this episode
        
        while world_state.is_mission_running and step < MAX_STEPS_PER_EPISODE:
            # Wait for valid video frame and observations
            timeout = 0
            while (len(world_state.video_frames) == 0 or len(world_state.observations) == 0) and world_state.is_mission_running and timeout < 20:
                time.sleep(0.05)
                world_state = agent_host.getWorldState()
                timeout += 1
            
            if timeout >= 20:
                print(f"Warning: Timeout waiting for observations at step {step}")
                break
            
            if not world_state.is_mission_running:
                print("Mission ended unexpectedly")
                break
                
            if len(world_state.video_frames) and len(world_state.observations):
                frame = world_state.video_frames[0]
                state = process_frame(frame)
                
                # Debug: Print frame info on first step of first episode
                if episode == 0 and step == 0:
                    print(f"Original frame size: {frame.height}x{frame.width}")
                    print(f"Processed state shape: {state.shape}")
                    print(f"Memory usage per frame: ~{state.nbytes} bytes")
                
                action = agent.select_action(state)
                action_name = ACTIONS[action]
                command, value = ACTION_COMMANDS[action_name]
                
                if step % 5 == 0:  # Print every 5 steps instead of every step
                    print(f"Step {step}: Selected action {action} ({action_name}: {command} {value})")
                
                # Only send turning commands (forward movement is continuous)
                agent_host.sendCommand(f"{command} {value}")
                
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
                
                # Check for finish line IMMEDIATELY after getting world state
                # This catches the emerald block reward before mission ends
                for reward_obj in world_state.rewards:
                    malmo_reward = reward_obj.getValue()
                    if malmo_reward > 5000:  # Emerald block reward
                        reached_finish_line = True
                        episode_completion_time = time.time() - episode_start_time
                        print(f"🏁 FINISH LINE REACHED! Agent touched emerald block!")
                        print(f"🎉 Episode {episode + 1} completed in {episode_completion_time:.2f} seconds!")
                        break
                
                # Wait for next frame and observations
                frame_timeout = 0
                while (len(world_state.video_frames) == 0 or len(world_state.observations) == 0) and world_state.is_mission_running and frame_timeout < 10:
                    time.sleep(0.05)
                    world_state = agent_host.getWorldState()
                    frame_timeout += 1
                
                if len(world_state.video_frames) and world_state.is_mission_running:
                    next_frame = world_state.video_frames[0]
                    next_state = process_frame(next_frame)
                    
                    reward, episode_end, current_z_pos, reached_finish = process_reward(world_state, step, previous_z_pos)
                    previous_z_pos = current_z_pos  # Update position for next step
                    
                    # Check if agent reached finish line (backup check in case we missed it above)
                    if reached_finish and not reached_finish_line:
                        reached_finish_line = True
                        episode_completion_time = time.time() - episode_start_time
                        print(f"🎉 Episode {episode + 1} completed in {episode_completion_time:.2f} seconds!")
                    
                    # Add action-specific reward
                    action_reward = get_action_reward(action)
                    reward += action_reward
                    
                    done = not world_state.is_mission_running or episode_end
                    agent.memory.push(state, action, reward, next_state, done)
                    agent.train()
                    
                    total_reward += reward
                    step += 1
                    
                    if step % 10 == 0:  # Print summary every 10 steps
                        print(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                    
                    if episode_end:
                        print("Episode ending due to mission completion or death")
                        break
                elif reached_finish_line:
                    # Mission ended due to reaching finish line, but we still want to process the final reward
                    # Create a dummy next state for the final transition
                    next_state = state  # Use current state as next state
                    reward, episode_end, current_z_pos, reached_finish = process_reward(world_state, step, previous_z_pos)
                    
                    # Add action-specific reward
                    action_reward = get_action_reward(action)
                    reward += action_reward
                    
                    done = True  # Mission is definitely done
                    agent.memory.push(state, action, reward, next_state, done)
                    agent.train()
                    
                    total_reward += reward
                    step += 1
                    
                    print(f"Final step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
                    break
                else:
                    print(f"Warning: No next frame available at step {step}")
                    break
            else:
                print(f"Warning: Missing video frame or observations at step {step}")
                break
        
            
        agent.metrics["rewards"].append(total_reward)
        print(f"\nEpisode {episode + 1} finished with total reward: {total_reward:.2f} after {step} steps")
        
        # If agent reached finish line, record the time
        if reached_finish_line:
            total_completions += 1
            
            # Calculate time improvement bonus before updating records: 100 points per 0.01 seconds
            time_improvement_bonus = 0
            if episode_completion_time < best_completion_time and best_completion_time != float('inf'):
                time_improvement_seconds = best_completion_time - episode_completion_time
                time_improvement_bonus = (time_improvement_seconds / 0.01) * 100
                print(f"🚀 NEW BEST TIME! Improved by {time_improvement_seconds:.3f} seconds")
                print(f"💰 Time improvement bonus: {time_improvement_bonus:.0f} points")
                
                # Add the bonus to total reward
                total_reward += time_improvement_bonus
                
                # Update the recorded reward in agent metrics to include the bonus
                agent.metrics["rewards"][-1] = total_reward
            elif best_completion_time == float('inf'):
                print(f"🎉 FIRST COMPLETION! No time bonus for inaugural run.")
            
            # Update the best completion time regardless of whether it's first or improved
            if episode_completion_time < best_completion_time:
                best_completion_time = episode_completion_time
            
            top_times = update_top_times(top_completion_times, episode + 1, episode_completion_time, total_reward)
            
            # Add completion times to agent metrics
            if "completion_times" not in agent.metrics:
                agent.metrics["completion_times"] = []
                agent.metrics["top_5_times"] = []
            
            agent.metrics["completion_times"].append({
                'episode': episode + 1,
                'time': episode_completion_time,
                'reward': total_reward
            })
            agent.metrics["top_5_times"] = top_times
            
            # Print current standings
            print_top_times(top_completion_times)
            print(f"Total completions so far: {total_completions}")
            
            # Save completion times to file after each completion
            file_path = save_completion_times_to_file(
                top_completion_times, 
                total_completions, 
                EPISODES, 
                completion_time=episode_completion_time, 
                episode=episode + 1, 
                reward=total_reward
            )
            if file_path:
                print(f"📝 Completion times saved to: {file_path}")
        
        # Stop forward movement
        agent_host.sendCommand("move 0")
        
        # Properly end the mission and wait
        print("Ending mission...")
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:
            agent_host.sendCommand("quit")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
        
        print("Waiting for mission to fully end...")
        time.sleep(1)  # Increased wait time for world reset
        
        # Clear memory and force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        # Checkpoint saving logic
        save_checkpoint(agent, episode + 1, total_reward, best_reward, best_completion_time, "latest")
        
        # 2. Save best checkpoint when we get a new best reward
        if total_reward > best_reward:
            best_reward = total_reward
            save_checkpoint(agent, episode + 1, total_reward, best_reward, best_completion_time, "best")

    # Save final model after all episodes complete
    print("Training completed! Saving final model...")
    try:
        # Save the final model using the agent's save_model method
        agent.save_model(EPISODES)
        print(f"Final model saved after {EPISODES} episodes")
        
        # Also save as latest checkpoint
        save_checkpoint(agent, EPISODES, agent.metrics["rewards"][-1] if agent.metrics["rewards"] else 0, best_reward, best_completion_time, "latest")
        
        # If the final episode was the best, save it as best too
        final_reward = agent.metrics["rewards"][-1] if agent.metrics["rewards"] else 0
        if final_reward >= best_reward:
            save_checkpoint(agent, EPISODES, final_reward, best_reward, best_completion_time, "best")
            
        print(f"Best reward achieved: {best_reward:.2f}")
        
        # Print final completion times summary
        if total_completions > 0:
            print(f"\n🏁 TRAINING SUMMARY 🏁")
            print(f"Total episodes completed: {total_completions}/{EPISODES}")
            print(f"Completion rate: {(total_completions/EPISODES)*100:.1f}%")
            print_top_times(top_completion_times)
            
            # Save final completion times to file
            final_file_path = save_completion_times_to_file(
                top_completion_times, 
                total_completions, 
                EPISODES
            )
            if final_file_path:
                print(f"📁 Final completion times report saved to: {final_file_path}")
        else:
            print("\nNo episodes were completed during training.")
            # Still save file even if no completions
            save_completion_times_to_file([], 0, EPISODES)
        
    except Exception as e:
        print(f"Error saving final model: {e}")
        print("Training metrics will still be available in agent.metrics")

if __name__ == "__main__":
    main() 