import MalmoPython
import time
import json
import numpy as np
from agent import Agent
import os
from pathlib import Path

# Constants
XML_FILE = "./envs/movement.xml"  # You'll need to create this XML file
EPISODES = 5000
MAX_STEPS_PER_EPISODE = 1000
INPUT_SHAPE = (84, 112, 3)  # Height, Width, Channels
SAVE_INTERVAL = 100
TARGET_UPDATE = 10

# Define actions (simplified movement only)
ACTIONS = [
    "move 1",    # Forward
    "move -1",   # Backward
    "strafe 1",  # Right
    "strafe -1", # Left
]

def create_mission_spec():
    xml = Path(XML_FILE).read_text()
    return MalmoPython.MissionSpec(xml, True)

def process_frame(frame):
    pixels = np.array(frame.pixels, dtype=np.uint8)
    frame_shape = (frame.height, frame.width, frame.channels)
    image = pixels.reshape(frame_shape)
    return image

def main():
    # Initialize agent
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
    
    # Initialize Malmo
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print("ERROR:", e)
        print(agent_host.getUsage())
        exit(1)
    
    agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
    agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
    
    # Training loop
    for episode in range(EPISODES):
        print(f"\nEpisode {episode + 1}/{EPISODES}")
        
        # Mission setup
        mission = create_mission_spec()
        record = MalmoPython.MissionRecordSpec()
        
        # Start mission
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
        
        # Wait for mission to start
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
        
        # Episode loop
        total_reward = 0
        step = 0
        
        while world_state.is_mission_running and step < MAX_STEPS_PER_EPISODE:
            # Get current state
            if len(world_state.video_frames):
                frame = world_state.video_frames[0]
                state = process_frame(frame)
                
                # Select and take action
                action = agent.select_action(state)
                agent_host.sendCommand(ACTIONS[action])
                
                # Get next state and reward
                time.sleep(0.1)  # Small delay to allow the action to take effect
                world_state = agent_host.getWorldState()
                
                if len(world_state.video_frames):
                    next_frame = world_state.video_frames[0]
                    next_state = process_frame(next_frame)
                    
                    # Process reward (you'll need to implement this based on your XML)
                    reward = 0
                    if len(world_state.observations):
                        obs = json.loads(world_state.observations[-1].text)
                        # Add your reward logic here based on observations
                    
                    # Store experience
                    agent.memory.push(state, action, reward, next_state, False)
                    
                    # Train
                    loss = agent.train()
                    
                    total_reward += reward
                    step += 1
                    
                    print(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}", end="\r")
        
        # End episode
        agent.metrics["rewards"].append(total_reward)
        print(f"\nEpisode {episode + 1} finished with total reward: {total_reward:.2f}")
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        # Save model
        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save_model(episode + 1)
            print(f"Model saved at episode {episode + 1}")

if __name__ == "__main__":
    main() 