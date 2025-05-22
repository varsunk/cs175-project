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
EPISODES = 5000
MAX_STEPS_PER_EPISODE = 1000
INPUT_SHAPE = (480, 640, 3)  # Height, Width, Channels from treasure_hunt.xml
SAVE_INTERVAL = 10
TARGET_UPDATE = 10

# Define actions (simplified movement only)
ACTIONS = [
    "move 1",     # Forward
    "move -1",    # Backward
    "strafe 1",   # Right
    "strafe -1",  # Left
    "turn 1",     # Turn right
    "turn -1",    # Turn left
]

# Reward constants
REWARDS = {
    "diamond": 10,
    "death": -200,
    "time_penalty": -0.01,
    "command": -1
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

def process_reward(world_state, step):
    reward = 0
    episode_end = False
    
    if len(world_state.observations):
        obs = json.loads(world_state.observations[-1].text)
        
        if "Inventory" in obs:
            inventory = obs["Inventory"]
            for item in inventory:
                if item["type"] == "diamond":
                    reward += REWARDS["diamond"]
                    print(f"Collected diamond! Reward: {REWARDS['diamond']}")
        
        if "Life" in obs and obs["Life"] <= 0:
            reward += REWARDS["death"]
            episode_end = True
            print("Agent died! Episode ending.")
    
    reward += REWARDS["time_penalty"]
    reward += REWARDS["command"]
    
    return reward, episode_end

def enter_boat(agent_host):
    """Attempt to enter the boat by right-clicking"""
    print("Attempting to enter boat...")
    agent_host.sendCommand("use 1")  # Right-click to enter boat
    time.sleep(1)  # Wait for boat entry animation
    agent_host.sendCommand("use 0")  # Release right-click
    
    # Check if agent is in boat
    world_state = agent_host.getWorldState()
    if len(world_state.observations):
        obs = json.loads(world_state.observations[-1].text)
        if "IsInWater" in obs and obs["IsInWater"] == 1:
            print("Successfully entered boat!")
        else:
            print("Failed to enter boat, trying again...")
            # Try one more time
            agent_host.sendCommand("use 1")
            time.sleep(1)
            agent_host.sendCommand("use 0")
    else:
        print("No observations available to verify boat entry")

def main():
    print("Initializing agent...")
    agent = Agent(
        input_shape=INPUT_SHAPE,
        n_actions=len(ACTIONS),
        alpha=0.0005,
        gamma=0.85,
        epsilon=1.0,
        epsilon_decay=0.992,
        epsilon_min=0.05,
        batch_size=64,
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
        
        # Enter boat at the start of each episode
        enter_boat(agent_host)
        
        total_reward = 0
        step = 0
        
        while world_state.is_mission_running and step < MAX_STEPS_PER_EPISODE:
            if len(world_state.video_frames):
                frame = world_state.video_frames[0]
                state = process_frame(frame)
                
                action = agent.select_action(state)
                command = ACTIONS[action]
                print(f"Step {step}: Selected action {action} ({command})")
                agent_host.sendCommand(command)
                
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
                
                if len(world_state.video_frames):
                    next_frame = world_state.video_frames[0]
                    next_state = process_frame(next_frame)
                    
                    reward, episode_end = process_reward(world_state, step)
                    
                    done = not world_state.is_mission_running or episode_end
                    agent.memory.push(state, action, reward, next_state, done)
                    
                    loss = agent.train()
                    
                    total_reward += reward
                    step += 1
                    
                    print(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                    
                    if episode_end:
                        print("Ending episode due to death")
                        agent_host.sendCommand("quit")
                        break
        
        agent.metrics["rewards"].append(total_reward)
        print(f"\nEpisode {episode + 1} finished with total reward: {total_reward:.2f}")
        
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save_model(episode + 1)
            print(f"Model saved at episode {episode + 1}")

if __name__ == "__main__":
    main() 