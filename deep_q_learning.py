import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
from GridEnv import GridEnv
from enum import Enum

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_MAPS = [f"map2.json" ]
TEST_MAP = "map2.json"

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        # transition = (state_grid, state_agent, action, reward, next_grid, next_agent, done)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, grid_size, n_channels, n_actions, agent_dim=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        conv_out_size = 32 * grid_size * grid_size
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + agent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, grid_obs, agent_pos):
        # grid_obs = (batch_size, n_channel, grid_size, grid_size)
        x = self.conv(grid_obs)  # (batch_size, 32, grid_size, grid_size)
        x = x.view(x.size(0), -1) # (batch_size, 32 * grid_size * grid_size)
        x = torch.cat([x, agent_pos], dim=1)   # (batch_size, 32 * grid_size * gird_size + agent_dim)
        return self.fc(x)  # (batch_size, n_actions)


class QLearning:
    def __init__(
        self,
        env,
        learning_rate=1e-3,
        epsilon=1.0,
        min_epsilon=0.05,
        discount_rate=0.99,
        training_episodes=100
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.discount_rate = discount_rate
        self.training_episodes = training_episodes

        sample_obs, _ = env.reset()
        self.grid_size = sample_obs['grid'].shape[0]
        self.n_channels = sample_obs['grid'].shape[2]
        self.n_actions = env.action_space.n

        self.policy_net = DQN(self.grid_size, self.n_channels, self.n_actions).to(DEVICE)
        self.target_net = DQN(self.grid_size, self.n_channels, self.n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(capacity=100)

        self.rewards_per_episode = []
        self.steps_per_episode = []
        self.epsilon_values = []
        self.successful_episodes = 0

    def preprocess(self, obs):
        """
        obs['grid'] : (grid_size, grid_size, 3)
        obs['agent'] : (2,)
        Output:
        grid : (1, 3, grid_size, grid_size)
        agent: (1,2)
        normalize the agent coordinates from (0, grid_size-1) to (0,1)
        """
        grid = torch.tensor(obs['grid'], dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)
        agent = torch.tensor(obs['agent'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        agent = agent / (self.grid_size - 1)
        return grid, agent

    def choose_action(self, grid_obs, agent_obs, training=True):
        """
        training = True when training
        training = False means pure greedy, for testing

        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q_vals = self.policy_net(grid_obs, agent_obs)
            return q_vals.argmax().item() # Choose the action with the hights value

    def optimize_model(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        grids, agents, actions, rewards, next_grids, next_agents, dones = zip(*transitions)
        grid_batch = torch.cat(grids) # (batch, grid_size, grid_size, 3)
        agent_batch = torch.cat(agents) # (batch,2)
        action_batch = torch.tensor(actions, device=DEVICE).unsqueeze(1)  # (batch, 1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32,device=DEVICE).unsqueeze(1) # (batch, 1)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1) # (batch,1)
        next_grid_batch = torch.cat(next_grids) # (batch, grid_size, grid_size, 3)
        next_agent_batch = torch.cat(next_agents) # (batch,2)

        q_values = self.policy_net(grid_batch, agent_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q = self.target_net(next_grid_batch, next_agent_batch).max(1)[0].unsqueeze(1)
            target = reward_batch + self.discount_rate * next_q * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _update_gui_text(self, episode, total_reward):
        self.env.set_episode_text(f"Episode: {episode+1}/{self.training_episodes}")
        self.env.set_success_text(f"Success: {self.successful_episodes}")
        self.env.set_info_text(f"Eps: {self.epsilon:.3f} | LR: {self.learning_rate}")
        self.env.set_reward_text(f"Reward: {total_reward:.1f}")
        self.env.render()

    def train(self):
        for ep in tqdm(range(self.training_episodes), desc="DQN Training"):
            # Load map
            map = TRAIN_MAPS[ep % len(TRAIN_MAPS)]
            self.env.load_map(map)

            obs, _ = self.env.reset()
            grid_obs, agent_obs = self.preprocess(obs)
            done = False
            total_reward = 0
            steps = 0
            success = False

            # Decay epsilon
            self.epsilon = max(self.min_epsilon,
                               self.min_epsilon + (self.epsilon - self.min_epsilon)*np.exp(-1e-3*ep))
            self.epsilon_values.append(self.epsilon)

            while not done and steps < 200:
                action = self.choose_action(grid_obs, agent_obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_grid, next_agent = self.preprocess(next_obs)

                self.memory.push((grid_obs, agent_obs,
                                  action, reward,
                                  next_grid, next_agent,
                                  float(terminated)))
                self.optimize_model()

                grid_obs, agent_obs = next_grid, next_agent
                total_reward += reward
                steps += 1

                if terminated:
                    success = True
                    break

            if success:
                self.successful_episodes += 1
            self.rewards_per_episode.append(total_reward)
            self.steps_per_episode.append(steps)

            # Update GUI
            self._update_gui_text(ep, total_reward)

            # Update target network
            if (ep + 1) % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def evaluate(self, num_episodes=20):
        self.policy_net.eval()
        self.env.load_map(TEST_MAP)
        rewards, successes = [], 0
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            grid_obs, agent_obs = self.preprocess(obs)
            done, total_r = False, 0
            while not done:
                action = self.choose_action(grid_obs, agent_obs, training=False)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                grid_obs, agent_obs = self.preprocess(next_obs)
                total_r += reward
                done = terminated
            rewards.append(total_r)
            if done:
                successes += 1
        print(f"Evaluation over {num_episodes} episodes:")
        print(f"Average Reward: {np.mean(rewards):.2f}")
        print(f"Success: {successes}")
        self.policy_net.train()

if __name__ == '__main__':
    env = GridEnv(render_mode='human', map_file=TRAIN_MAPS[0])
    agent = QLearning(env,
                      learning_rate=1e-3,
                      epsilon=1.0,
                      min_epsilon=0.1,
                      discount_rate=0.99,
                      training_episodes=300)
    # agent.train()
    # # save trained weights
    # torch.save(agent.policy_net.state_dict(), 'dqn_policy2.pth')

    # load weights and evaluate
    agent.policy_net.load_state_dict(torch.load('dqn_policy2.pth', map_location=DEVICE))
    agent.evaluate(num_episodes=50)
    env.close()
