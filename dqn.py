import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        # input_shape is (height, width, channels) = (480, 640, 3)
        # For Conv2d we need (channels, height, width)
        channels, height, width = input_shape[2], input_shape[0], input_shape[1]
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the flattened features
        self._to_linear = None
        self._get_conv_output(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def _get_conv_output(self, shape):
        # Create a dummy input to calculate the size of the flattened features
        # Input shape is (height, width, channels), convert to (batch, channels, height, width)
        x = torch.zeros(1, shape[2], shape[0], shape[1])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        self._to_linear = x.numel() // x.shape[0]  # Total elements per batch
        
    def forward(self, x):
        # Input x should be (batch_size, channels, height, width)
        # If input is (channels, height, width), add batch dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x) 