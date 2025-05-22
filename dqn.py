import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        
        # Calculate the size of the flattened features
        self._to_linear = None
        self._get_conv_output(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, n_actions)
        
    def _get_conv_output(self, shape):
        # Create a dummy input to calculate the size of the flattened features
        x = torch.zeros(1, shape[2], shape[0], shape[1])
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        
    def forward(self, x):
        # Ensure input is in the correct format (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) 