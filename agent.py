import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN
from replay_buffer import ReplayBuffer
import json
import os

class Agent:
    def __init__(self, input_shape, n_actions, alpha=0.0005, gamma=0.85, epsilon=1.0,
                 epsilon_decay=0.992, epsilon_min=0.05, batch_size=64, mem_size=5000,
                 model_dir="models", metrics_dir="metrics"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Initialize DQN and target networks
        self.policy_net = DQN(input_shape, n_actions).to(self.device)
        self.target_net = DQN(input_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.memory = ReplayBuffer(mem_size)
        
        # Metrics tracking
        self.model_dir = model_dir
        self.metrics_dir = metrics_dir
        self.metrics = {
            "rewards": [],
            "losses": [],
            "epsilons": []
        }
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update metrics
        self.metrics["losses"].append(loss.item())
        self.metrics["epsilons"].append(self.epsilon)
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, episode):
        model_path = os.path.join(self.model_dir, f"model_episode_{episode}.pth")
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, model_path)
        
        metrics_path = os.path.join(self.metrics_dir, f"metrics_episode_{episode}.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f)
    
    def load_model(self, episode):
        model_path = os.path.join(self.model_dir, f"model_episode_{episode}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            
            metrics_path = os.path.join(self.metrics_dir, f"metrics_episode_{episode}.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f) 