import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from enum import Enum
from GridEnv import GridEnv
import pygame
import matplotlib

matplotlib.use('Agg')


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class QLearning:
    def __init__(self, env, learning_rate=0.1, epsilon=1.0,
                 min_epsilon=0.01, discount_rate=0.99,
                 training_episodes=100):
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.discount_rate = discount_rate
        self.training_episodes = training_episodes

        # Change: initialize q_table as [state_space, actions]
        self.q_table = np.zeros((env.rows * env.cols, len(Actions)))

        # Tracking metrics
        self.rewards_per_episode = []
        self.steps_per_episode = []
        self.epsilon_values = []
        self.successful_episodes = 0

    def choose_action(self, state, training=True):
        agent_pos = state["agent"]
        # Convert 2D position to state index
        state_index = agent_pos[0] * self.env.cols + agent_pos[1]
        if training and random.random() < self.epsilon:
            return random.randint(0, len(Actions) - 1)  # Random action
        else:
            return np.argmax(self.q_table[state_index])  # Greedy action

    def train(self):
        self.successful_episodes = 0
        for episode in tqdm(range(self.training_episodes), desc="Training"):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            episode_success = False
            # Update text
            self._update_gui_text(episode)

            # Decay epsilon
            self.epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-0.001 * episode)
            self.epsilon_values.append(self.epsilon)

            while not done and steps < 100:
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Convert positions to state indices
                current_pos = state["agent"]
                next_pos = next_state["agent"]
                current_index = current_pos[0] * self.env.cols + current_pos[1]
                next_index = next_pos[0] * self.env.cols + next_pos[1]

                # Q-learning update
                best_next_action = np.argmax(self.q_table[next_index])
                td_target = reward + self.discount_rate * self.q_table[next_index, best_next_action]
                self.q_table[current_index, action] += self.learning_rate * (
                    td_target - self.q_table[current_index, action]
                )

                # Update tracking
                total_reward += reward
                steps += 1
                state = next_state

                # Check if reached goal
                if terminated:
                    episode_success = True
                    done = True
                    break

            if episode_success:
                self.successful_episodes += 1

            self.rewards_per_episode.append(total_reward)
            self.steps_per_episode.append(steps)

        return self.q_table

    def _update_gui_text(self, episode):
        self.env.set_episode_text(f"Episode: {episode + 1}/{self.training_episodes}")
        self.env.set_success_text(f"Successful episodes: {self.successful_episodes}")
        self.env.set_info_text(f"Epsilon: {self.epsilon:.3f} | Learning Rate: {self.learning_rate}")
        self.env.render()



if __name__ == "__main__":

    env = GridEnv(render_mode='human', map_file='map2.json')
    q_learner = QLearning(env, training_episodes=100, epsilon=1)
    q_learner.train()

    # Visualize and evaluate using the modified Q-table representation
    env.get_policy_visualization(q_learner.q_table)


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.time.delay(100)

    env.close()