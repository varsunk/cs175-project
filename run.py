import MalmoPython # type: ignore
import time
import json
import numpy as np
from agent import Agent
import os
from pathlib import Path
import sys
import torch

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
EPISODES = 500
MAX_STEPS_PER_EPISODE = 100  # Reduced for faster testing
INPUT_SHAPE = (480, 640, 3)  # Height, Width, Channels from treasure_hunt.xml
SAVE_INTERVAL = 10
TARGET_UPDATE = 10

# Define actions (only turning to navigate the race track)
ACTIONS = [
    "turn_right",      # Turn right
    "turn_left",       # Turn left
    "no_turn",         # Continue straight (no turning)
]

# Action mapping to continuous movement commands
ACTION_COMMANDS = {
    "turn_right": ("turn", 0.5),
    "turn_left": ("turn", -0.5),
    "no_turn": ("turn", 0),
}

# Reward constants
REWARDS = {
    "death": -50,               # Death penalty
    "time_penalty": -0.001,     # Very small time penalty to encourage efficiency
    "movement_bonus": 0.01,     # Small bonus for forward movement
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
    # Transpose from (H, W, C) to (C, H, W) for PyTorch
    image = np.transpose(image, (2, 0, 1))
    return image

def process_reward(world_state, step, prev_z_pos):
    reward = 0
    episode_end = False
    current_z_pos = prev_z_pos  # Keep for compatibility
    
    # Get rewards from Malmo's reward system
    for reward_obj in world_state.rewards:
        malmo_reward = reward_obj.getValue()
        reward += malmo_reward
        print(f"Malmo reward received: {malmo_reward}")
        
        # Check if this is a finish line reward (emerald block)
        if malmo_reward >= 500:
            episode_end = True
            print(f"Finish line reached! Episode ending with reward: {malmo_reward}")
    
    if len(world_state.observations):
        obs = json.loads(world_state.observations[-1].text)
        
        # Check for death
        if "Life" in obs and obs["Life"] <= 0:
            reward += REWARDS["death"]
            episode_end = True
            print("Agent died! Episode ending.")
        
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
    
    return reward, episode_end, current_z_pos

def get_action_reward(action):
    """Give bonus reward for forward movement and penalize excessive turning"""
    action_name = ACTIONS[action]
    if action_name == "no_turn":  # Going straight - encourage this
        return REWARDS["movement_bonus"]
    elif action_name in ["turn_right", "turn_left"]:  # Turning - small penalty to discourage unnecessary turns
        return -0.005
    else:
        return 0.0

def main():
    print("Initializing agent...")
    agent = Agent(
        input_shape=INPUT_SHAPE,
        n_actions=len(ACTIONS),
        alpha=0.001,           # Slightly higher learning rate
        gamma=0.90,            # Higher discount factor for longer-term planning
        epsilon=0.7,           # Start with lower epsilon (less random)
        epsilon_decay=0.995,   # Slower decay
        epsilon_min=0.1,       # Higher minimum epsilon to maintain some exploration
        batch_size=32,         # Smaller batch size for more frequent updates
        mem_size=5000
    )
    
    print("Initializing Malmo...")
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print("ERROR:", e)
        print(agent_host.getUsage())
        exit(1)
    
    agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
    agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
    
    for episode in range(EPISODES):
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
                
                action = agent.select_action(state)
                action_name = ACTIONS[action]
                command, value = ACTION_COMMANDS[action_name]
                
                if step % 5 == 0:  # Print every 5 steps instead of every step
                    print(f"Step {step}: Selected action {action} ({action_name}: {command} {value})")
                
                # Only send turning commands (forward movement is continuous)
                agent_host.sendCommand(f"{command} {value}")
                
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
                
                # Wait for next frame and observations
                frame_timeout = 0
                while (len(world_state.video_frames) == 0 or len(world_state.observations) == 0) and world_state.is_mission_running and frame_timeout < 10:
                    time.sleep(0.05)
                    world_state = agent_host.getWorldState()
                    frame_timeout += 1
                
                if len(world_state.video_frames) and world_state.is_mission_running:
                    next_frame = world_state.video_frames[0]
                    next_state = process_frame(next_frame)
                    
                    reward, episode_end, current_z_pos = process_reward(world_state, step, previous_z_pos)
                    previous_z_pos = current_z_pos  # Update position for next step
                    
                    # Add action-specific reward
                    action_reward = get_action_reward(action)
                    reward += action_reward
                    
                    done = not world_state.is_mission_running or episode_end
                    agent.memory.push(state, action, reward, next_state, done)
                    
                    loss = agent.train()
                    
                    total_reward += reward
                    step += 1
                    
                    if step % 10 == 0:  # Print summary every 10 steps
                        print(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                    
                    if episode_end:
                        print("Episode ending due to mission completion or death")
                        break
                else:
                    print(f"Warning: No next frame available at step {step}")
                    break
            else:
                print(f"Warning: Missing video frame or observations at step {step}")
                break
        
        # Check why episode ended
        if step == 0:
            print("Warning: Episode ended immediately - check mission setup")
        elif step < 5:
            print(f"Warning: Very short episode ({step} steps) - mission may be ending too quickly")
            
        agent.metrics["rewards"].append(total_reward)
        print(f"\nEpisode {episode + 1} finished with total reward: {total_reward:.2f} after {step} steps")
        
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
        
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        # Commented out frequent saving to avoid disk space/corruption issues
        # if (episode + 1) % SAVE_INTERVAL == 0:
        #     agent.save_model(episode + 1)
        #     print(f"Model saved at episode {episode + 1}")

    # Save final model after all episodes complete
    print("Training completed! Saving final model...")
    try:
        agent.save_model(EPISODES)
        print(f"Final model saved after {EPISODES} episodes")
    except Exception as e:
        print(f"Error saving final model: {e}")
        print("Training metrics will still be available in agent.metrics")

if __name__ == "__main__":
    main() 