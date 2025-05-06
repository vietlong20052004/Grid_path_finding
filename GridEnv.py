import pygame
import numpy as np
import json
from typing import Dict, Tuple, Optional, List
import gymnasium as gym
from gymnasium import spaces
from enum import Enum

from matplotlib import pyplot as plt


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, grid_size=10, map_file=None):
        # Environment configuration
        self.grid_size = grid_size
        self.rows = grid_size
        self.cols = grid_size
        self.cell_size = 60
        self.window_size = self.grid_size * self.cell_size

        self.load_map(map_file)

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=int),
            "grid": spaces.Box(low=-1, high=1, shape=(self.grid_size, self.grid_size, 3), dtype=np.float32)
        })

        self.action_space = spaces.Discrete(4)

        # Movement mapping
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([0, 1]),
            Actions.UP.value: np.array([-1,0]),
            Actions.LEFT.value: np.array([0, -1]),
            Actions.DOWN.value: np.array([1, 0]),
        }

        # Rendering setup
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None

        # Colors
        self.colors = {
            'background': (255, 255, 255),
            'grid': (200, 200, 200),
            'obstacle': (0, 0, 0),
            'start': (0, 0, 255),
            'goal': (255, 215, 0),
            'agent': (128, 0, 128),
            'reward_pos': (0, 255, 0),
            'reward_neg': (255, 0, 0),
            'text': (0, 0, 0)
        }
        self.info_text = ""
        self.episode_text = ""
        self.success_text = ""
        self.reward_text = ""

    def load_map(self, file_path: str):
        """Load map configuration from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.grid_size = data['rows']
        self.window_size = self.grid_size * self.cell_size

        # Initialize grid and obstacles
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Set obstacles and rewards
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = data['grid'][row][col]
                self.obstacles[row, col] = cell['obstacle']
                self.grid[row, col] = cell['reward']

        # Set start and goal positions
        self.start_pos = np.array(data['start_pos'])
        self.goal_pos = np.array(data['goal_pos'])

    def _get_obs(self):
        """Get current observation"""
        grid_obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)

        # Channel 0: Obstacles (-1) and free space (0)
        grid_obs[:, :, 0] = self.obstacles.astype(np.float32)

        # Channel 1: Positive rewards (value) and 0 otherwise
        grid_obs[:, :, 1] = np.where(self.grid > 0, self.grid, 0)

        # Channel 2: Negative rewards (absolute value) and 0 otherwise
        grid_obs[:, :, 2] = np.where(self.grid < 0, -self.grid, 0)

        return {
            "agent": self._agent_location.copy(),
            "grid": grid_obs
        }

    def _get_info(self):
        """Get auxiliary information"""
        return {
            "distance": np.linalg.norm(self._agent_location - self.goal_pos, ord=1),
            "reward": self.grid[tuple(self._agent_location)]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent position
        self._agent_location = self.start_pos.copy()

        # Reset goal position
        self._target_location = self.goal_pos.copy()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def set_info_text(self, text):
        self.info_text = text

    def set_episode_text(self, text):
        self.episode_text = text

    def set_success_text(self, text):
        self.success_text = text

    def set_reward_text(self, text):
        self.reward_text = text

    def step(self, action):
        direction = self._action_to_direction[action]
        new_location = self._agent_location.copy() + direction
        reward = 0
        terminated = False
        truncated = False

        # Check boundaries
        if (0 <= new_location[0] < self.grid_size and
                0 <= new_location[1] < self.grid_size):

            # Check obstacles
            if not self.obstacles[tuple(new_location)]:
                self._agent_location = new_location
                reward += self.grid[tuple(new_location)]
            else:
                reward += self.grid[tuple(new_location)]
                return self._get_obs(), reward, terminated, truncated, self._get_info()
        else:
            reward -= 20
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        # Check if reached goal
        terminated = np.array_equal(self._agent_location, self.goal_pos)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Grid Path Planning")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 20)
            self.info_font = pygame.font.SysFont('Arial', 16)

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.colors['background'])

        # Draw grid lines
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(canvas, self.colors['grid'], (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(canvas, self.colors['grid'], (0, y), (self.window_size, y))

        # Draw cells
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                # Draw obstacles
                if self.obstacles[row, col]:
                    pygame.draw.rect(canvas, self.colors['obstacle'], rect)
                    if self.grid[row, col] != 0:
                        text = self.font.render(f"{self.grid[row, col]:.1f}", True, (255, 255, 255))
                        canvas.blit(text, text.get_rect(center=rect.center))

                # Draw rewards (non-obstacles)
                elif self.grid[row, col] != 0:
                    color = self.colors['reward_pos'] if self.grid[row, col] > 0 else self.colors['reward_neg']
                    pygame.draw.rect(canvas, color, rect, 3)
                    text = self.font.render(f"{self.grid[row, col]:.1f}", True, self.colors['text'])
                    canvas.blit(text, text.get_rect(center=rect.center))

        # Draw start position
        start_rect = pygame.Rect(
            self.start_pos[1] * self.cell_size,
            self.start_pos[0] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(canvas, self.colors['start'], start_rect)
        text = self.font.render("S", True, (255, 255, 255))
        canvas.blit(text, text.get_rect(center=start_rect.center))

        # Draw goal position
        goal_rect = pygame.Rect(
            self.goal_pos[1] * self.cell_size,
            self.goal_pos[0] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(canvas, self.colors['goal'], goal_rect)
        text = self.font.render("G", True, (0, 0, 0))
        canvas.blit(text, text.get_rect(center=goal_rect.center))

        # Draw agent
        agent_center = (
            self._agent_location[1] * self.cell_size + self.cell_size // 2,
            self._agent_location[0] * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(
            canvas,
            self.colors['agent'],
            agent_center,
            self.cell_size // 3
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())

            # Create a semi-transparent overlay for text
            text_overlay = pygame.Surface((self.window_size, 80), pygame.SRCALPHA)
            text_overlay.fill((0, 0, 0, 150))  # Black with 70% opacity

            # Blit the overlay at the top
            self.window.blit(text_overlay, (0, 0))

            # Render and display the text information
            episode_surface = self.info_font.render(self.episode_text, True, (255, 255, 255))
            success_surface = self.info_font.render(self.success_text, True, (255, 255, 255))
            info_surface = self.info_font.render(self.info_text, True, (255, 255, 255))

            # Position the text with some padding
            self.window.blit(episode_surface, (10, 10))
            self.window.blit(success_surface, (10, 30))
            self.window.blit(info_surface, (10, 50))

            # Add a separator line
            pygame.draw.line(self.window, (100, 100, 100), (0, 80), (self.window_size, 80), 1)

            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def get_policy_visualization(self, q_table):
        """Generate a visualization of the policy from Q-table using arrow images"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Policy Visualization")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 20)

        # Load arrow images with error handling
        arrows_loaded = False
        arrow_images = {}
        try:
            arrow_size = self.cell_size // 2  # Size for the arrow images

            # Load and scale each arrow image
            arrow_images[Actions.RIGHT.value] = pygame.transform.scale(
                pygame.image.load("right_arrow.png").convert_alpha(),
                (arrow_size, arrow_size)
            )
            arrow_images[Actions.UP.value] = pygame.transform.scale(
                pygame.image.load("up_arrow.png").convert_alpha(),
                (arrow_size, arrow_size)
            )
            arrow_images[Actions.LEFT.value] = pygame.transform.scale(
                pygame.image.load("left_arrow.png").convert_alpha(),
                (arrow_size, arrow_size)
            )
            arrow_images[Actions.DOWN.value] = pygame.transform.scale(
                pygame.image.load("down_arrow.png").convert_alpha(),
                (arrow_size, arrow_size)
            )
            arrows_loaded = True
        except Exception as e:
            print(f"Error loading arrow images: {e}")
            arrows_loaded = False

        # Create canvas
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw grid lines
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(canvas, (230, 230, 230), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(canvas, (230, 230, 230), (0, y), (self.window_size, y))

        # Draw obstacles and special positions
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                # Draw obstacles
                if self.obstacles[row, col]:
                    pygame.draw.rect(canvas, (100, 100, 100), rect)
                    continue

                # Draw start position
                if (row, col) == tuple(self.start_pos):
                    pygame.draw.rect(canvas, (70, 130, 180), rect)
                    text = self.font.render("S", True, (255, 255, 255))
                    canvas.blit(text, text.get_rect(center=rect.center))
                    continue

                # Draw goal position
                if (row, col) == tuple(self.goal_pos):
                    pygame.draw.rect(canvas, (255, 215, 0), rect)
                    text = self.font.render("G", True, (0, 0, 0))
                    canvas.blit(text, text.get_rect(center=rect.center))
                    continue

        # Draw policy arrows
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Skip obstacles and special positions
                if (self.obstacles[row, col] or
                        (row, col) == tuple(self.start_pos) or
                        (row, col) == tuple(self.goal_pos)):
                    continue

                state_index = row * self.cols + col
                best_action = np.argmax(q_table[state_index])

                center_x = col * self.cell_size + self.cell_size // 2
                center_y = row * self.cell_size + self.cell_size // 2

                if arrows_loaded:
                    try:
                        # Get the appropriate arrow image
                        arrow_img = arrow_images[best_action]
                        # Calculate position to center the arrow
                        arrow_rect = arrow_img.get_rect(center=(center_x, center_y))
                        # Draw the arrow
                        canvas.blit(arrow_img, arrow_rect)
                    except Exception as e:
                        print(f"Error drawing arrow: {e}")
                        arrows_loaded = False  # Fall back to drawn arrows
                else:
                    # Fallback to drawing arrows
                    self._draw_arrow(canvas, center_x, center_y, best_action)

        # Add title
        title_font = pygame.font.SysFont('Arial', 24, bold=True)
        title_surface = title_font.render("Optimal Policy", True, (50, 50, 50))
        canvas.blit(title_surface, (self.window_size // 2 - title_surface.get_width() // 2, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.update()
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2)
            )

    def _draw_arrow(self, surface, x, y, direction):
        """Helper method to draw an arrow when images aren't available"""
        arrow_size = self.cell_size // 3
        arrow_color = (0, 100, 200)
        head_size = arrow_size // 2

        if direction == Actions.RIGHT.value:
            end_x = x + arrow_size
            pygame.draw.line(surface, arrow_color, (x, y), (end_x, y), 4)
            pygame.draw.polygon(surface, arrow_color, [
                (end_x, y),
                (end_x - head_size, y - head_size),
                (end_x - head_size, y + head_size)
            ])
        elif direction == Actions.UP.value:
            end_y = y - arrow_size
            pygame.draw.line(surface, arrow_color, (x, y), (x, end_y), 4)
            pygame.draw.polygon(surface, arrow_color, [
                (x, end_y),
                (x - head_size, end_y + head_size),
                (x + head_size, end_y + head_size)
            ])
        elif direction == Actions.LEFT.value:
            end_x = x - arrow_size
            pygame.draw.line(surface, arrow_color, (x, y), (end_x, y), 4)
            pygame.draw.polygon(surface, arrow_color, [
                (end_x, y),
                (end_x + head_size, y - head_size),
                (end_x + head_size, y + head_size)
            ])
        elif direction == Actions.DOWN.value:
            end_y = y + arrow_size
            pygame.draw.line(surface, arrow_color, (x, y), (x, end_y), 4)
            pygame.draw.polygon(surface, arrow_color, [
                (x, end_y),
                (x - head_size, end_y - head_size),
                (x + head_size, end_y - head_size)
            ])

    def save_policy_visualization(self, q_table, filename):
        """Save the policy visualization to an image file"""
        rgb_array = self.get_policy_visualization(q_table)
        if rgb_array is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_array)
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()