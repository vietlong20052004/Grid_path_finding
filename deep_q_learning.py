import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from tqdm import tqdm
import numpy as np
import pygame
from GridEnv import GridEnv


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        conv_out_size = int(np.prod(self.conv(torch.zeros(1,*input_shape)).size()))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 50),
            nn.ReLU(),
            nn.Linear(50,num_actions)
        )

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size()[0],-1))

    class Agent:
        def __init__(self, env, learning_rate=1.0, epsilon=1.0, min_epsilon=0.01, discount_rate=0.99,
                     training_episodes=100, batch_size=32, replay_memory_size=1000, sync_freq=20):
            self.env = env
            self.learning_rate = learning_rate
            self.epsilon = epsilon
            self.min_epsilon = min_epsilon
            self.discount_rate = discount_rate
            self.training_episodes = training_episodes
            self.batch_size = batch_size
            self.replay_memory = deque(maxlen=replay_memory_size)
            self.target_update_freq = target_update_freq
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            grid_shape = env.observation_space.spaces["grid"].shape  # (H, W, C)
            input_shape = (grid_shape[2], grid_shape[0], grid_shape[1])  # Convert to (C, H, W)
            self.num_actions = env.action_space.n