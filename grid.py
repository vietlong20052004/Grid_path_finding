import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import json
import random
from typing import Dict, Tuple, List, Optional


class GridEnvironment:
    """Core class representing the grid environment with obstacles and rewards."""

    def __init__(self, rows: int = 10, cols: int = 10):
        self.rows = rows
        self.cols = cols
        self.start_pos = None
        self.goal_pos = None
        self.grid = None
        self.reset_environment()



    def reset_environment(self) -> None:
        """Initialize or reset the environment with empty cells."""
        self.grid = [[{'obstacle': False, 'reward': 0.0} for _ in range(self.cols)]
                     for _ in range(self.rows)]
        self.start_pos = (0, 0)
        self.goal_pos = (self.rows - 1, self.cols - 1)

    def toggle_obstacle(self, row: int, col: int) -> None:
        """Toggle obstacle status of a cell."""
        if self.is_valid_position(row, col) and (row, col) != self.goal_pos:
            self.grid[row][col]['obstacle'] = not self.grid[row][col]['obstacle']

    def set_reward(self, row: int, col: int, reward: float) -> None:
        """Set reward value for a cell."""
        if self.is_valid_position(row, col):
            self.grid[row][col]['reward'] = reward

    def set_start(self, row: int, col: int) -> None:
        """Set the start position if the cell is not an obstacle."""
        if self.is_valid_position(row, col) and not self.grid[row][col]['obstacle']:
            self.start_pos = (row, col)

    def set_goal(self, row: int, col: int) -> None:
        """Set the goal position if the cell is not an obstacle."""
        if self.is_valid_position(row, col) and not self.grid[row][col]['obstacle']:
            self.goal_pos = (row, col)


    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def generate_random_map(self, obstacle_prob: float = 0.2,
                            reward_prob: float = 0.1,
                            min_reward: float = -1.0,
                            max_reward: float = 1.0) -> None:
        """Generate a random map with obstacles and rewards."""
        for row in range(self.rows):
            for col in range(self.cols):
                # Skip start and goal positions
                if (row, col) == self.start_pos or (row, col) == self.goal_pos:
                    continue

                # Set obstacles
                if random.random() < obstacle_prob:
                    self.grid[row][col]['obstacle'] = True
                else:
                    self.grid[row][col]['obstacle'] = False

                # Set rewards
                if random.random() < reward_prob:
                    self.grid[row][col]['reward'] = random.uniform(min_reward, max_reward)
                else:
                    self.grid[row][col]['reward'] = 0.0

    def to_dict(self) -> Dict:
        """Convert environment to a dictionary for serialization."""
        return {
            'rows': self.rows,
            'cols': self.cols,
            'grid': self.grid,
            'start_pos': self.start_pos,
            'goal_pos': self.goal_pos
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'GridEnvironment':
        """Create environment from a dictionary."""
        env = cls(data['rows'], data['cols'])
        env.grid = data['grid']
        env.start_pos = tuple(data['start_pos'])
        env.goal_pos = tuple(data['goal_pos'])
        return env

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)



class GridEnvironmentGUI:
    """Tkinter-based GUI for the grid environment."""

    def __init__(self, root: tk.Tk, env: GridEnvironment):
        self.root = root
        self.env = env
        self.cell_size = 40
        self.mode = 'obstacle'  # 'obstacle', 'reward', 'start', 'goal'
        self.setup_ui()

    def setup_ui(self) -> None:
        """Initialize all UI components."""
        self.root.title("Grid Environment")

        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Mode selection
        mode_frame = tk.LabelFrame(control_frame, text="Edit Mode")
        mode_frame.pack(side=tk.LEFT, padx=5)

        self.mode_var = tk.StringVar(value=self.mode)
        tk.Radiobutton(mode_frame, text="Obstacles", variable=self.mode_var,
                       value='obstacle', command=self.set_mode).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="Rewards", variable=self.mode_var,
                       value='reward', command=self.set_mode).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="Start", variable=self.mode_var,
                       value='start', command=self.set_mode).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="Goal", variable=self.mode_var,
                       value='goal', command=self.set_mode).pack(anchor=tk.W)

        # Environment controls
        env_frame = tk.LabelFrame(control_frame, text="Environment")
        env_frame.pack(side=tk.LEFT, padx=5)

        tk.Button(env_frame, text="New Grid", command=self.new_grid_dialog).pack(fill=tk.X)
        tk.Button(env_frame, text="Random Map", command=self.generate_random_map).pack(fill=tk.X)
        tk.Button(env_frame, text="Clear All", command=self.clear_environment).pack(fill=tk.X)

        # Bulk edit controls
        bulk_frame = tk.LabelFrame(control_frame, text="Bulk Edit")
        bulk_frame.pack(side=tk.LEFT, padx=5)

        tk.Button(bulk_frame, text="Set All Rewards",
                  command=self.set_all_rewards_dialog).pack(fill=tk.X)
        tk.Button(bulk_frame, text="Set Obstacle Rewards",
                  command=self.set_obstacle_rewards_dialog).pack(fill=tk.X)

        # File operations
        file_frame = tk.LabelFrame(control_frame, text="File")
        file_frame.pack(side=tk.LEFT, padx=5)

        tk.Button(file_frame, text="Save", command=self.save_environment).pack(fill=tk.X)
        tk.Button(file_frame, text="Load", command=self.load_environment).pack(fill=tk.X)

        # Canvas for grid
        self.canvas = tk.Canvas(
            self.root,
            width=self.env.cols * self.cell_size,
            height=self.env.rows * self.cell_size,
            bg='white'
        )
        self.canvas.pack(side=tk.TOP, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_cell_click)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(f"Mode: {self.mode} | Click on grid to edit")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.draw_grid()

    def set_mode(self) -> None:
        """Set the current edit mode."""
        self.mode = self.mode_var.get()
        self.status_var.set(f"Mode: {self.mode} | Click on grid to edit")

    def draw_grid(self) -> None:
        """Draw the grid on canvas with current environment state."""
        self.canvas.delete("all")

        # Draw grid lines
        for row in range(self.env.rows + 1):
            self.canvas.create_line(
                0, row * self.cell_size,
                   self.env.cols * self.cell_size, row * self.cell_size
            )

        for col in range(self.env.cols + 1):
            self.canvas.create_line(
                col * self.cell_size, 0,
                col * self.cell_size, self.env.rows * self.cell_size
            )

        # Draw cells
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                cell = self.env.grid[row][col]

                # Draw obstacles
                if cell['obstacle']:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='black')

                    # Visual indication for negative rewards
                    if cell['reward'] < 0:
                        # Add red diagonal stripes
                        self.canvas.create_line(x1, y1, x2, y2, fill='red', width=2)
                        self.canvas.create_line(x1, y2, x2, y1, fill='red', width=2)

                    # Display reward value if not zero
                    if cell['reward'] != 0:
                        text_color = 'white' if cell['reward'] < 0 else 'yellow'
                        self.canvas.create_text(
                            (x1 + x2) // 2, (y1 + y2) // 2,
                            text=f"{cell['reward']:.1f}",
                            fill=text_color,
                            font=('Arial', 9, 'bold')
                        )
                else:
                    # Draw normal cells with rewards
                    if cell['reward'] != 0:
                        color = 'green' if cell['reward'] > 0 else 'red'
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, stipple='gray25')
                        self.canvas.create_text(
                            (x1 + x2) // 2, (y1 + y2) // 2,
                            text=f"{cell['reward']:.1f}",
                            fill='black'
                        )

                # Draw start position
                if (row, col) == self.env.start_pos:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='blue')
                    self.canvas.create_text(
                        (x1 + x2) // 2, (y1 + y2) // 2,
                        text="Start",
                        fill='white'
                    )

                # Draw goal position
                if (row, col) == self.env.goal_pos:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='gold')
                    self.canvas.create_text(
                        (x1 + x2) // 2, (y1 + y2) // 2,
                        text="Goal",
                        fill='black'
                    )

    def set_all_rewards_dialog(self):
        """Set reward for all non-obstacle tiles"""
        reward = simpledialog.askfloat(
            "Set Rewards",
            "Enter reward value for all non-obstacle tiles:",
            parent=self.root
        )
        if reward is not None:
            for row in range(self.env.rows):
                for col in range(self.env.cols):
                    if not self.env.grid[row][col]['obstacle'] and (row, col) != self.env.goal_pos:
                        self.env.grid[row][col]['reward'] = reward
            self.draw_grid()

    def set_obstacle_rewards_dialog(self):
        """Set reward for all obstacle tiles"""
        reward = simpledialog.askfloat(
            "Set Obstacle Rewards",
            "Enter reward value for all obstacle tiles:",
            parent=self.root
        )
        if reward is not None:
            for row in range(self.env.rows):
                for col in range(self.env.cols):
                    if self.env.grid[row][col]['obstacle'] and (row, col) != self.env.goal_pos:
                        self.env.grid[row][col]['reward'] = reward
            self.draw_grid()


    def on_cell_click(self, event) -> None:
        """Handle cell click events based on current mode."""
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if not self.env.is_valid_position(row, col):
            return

        if self.mode == 'obstacle':
            self.env.toggle_obstacle(row, col)
            # Ensure start/goal positions aren't obstacles
            if (row, col) == self.env.start_pos and self.env.grid[row][col]['obstacle']:
                self.env.start_pos = (0, 0)
            if (row, col) == self.env.goal_pos and self.env.grid[row][col]['obstacle']:
                self.env.goal_pos = (self.env.rows - 1, self.env.cols - 1)
        elif self.mode == 'reward':
            reward = simpledialog.askfloat(
                "Set Reward",
                f"Enter reward value for cell ({row}, {col}):",
                parent=self.root
            )
            if reward is not None:
                self.env.set_reward(row, col, reward)
        elif self.mode == 'start':
            if not self.env.grid[row][col]['obstacle']:
                self.env.set_start(row, col)
            else:
                messagebox.showerror("Error", "Cannot place start position on an obstacle!")
        elif self.mode == 'goal':
            if not self.env.grid[row][col]['obstacle']:
                self.env.set_goal(row, col)
            else:
                messagebox.showerror("Error", "Cannot place goal position on an obstacle!")

        self.draw_grid()

    def new_grid_dialog(self) -> None:
        """Show dialog to create new grid with custom dimensions."""
        rows = simpledialog.askinteger(
            "New Grid",
            "Enter number of rows:",
            parent=self.root,
            minvalue=1,
            maxvalue=50,
            initialvalue=self.env.rows
        )

        cols = simpledialog.askinteger(
            "New Grid",
            "Enter number of columns:",
            parent=self.root,
            minvalue=1,
            maxvalue=50,
            initialvalue=self.env.cols
        )

        if rows and cols:
            self.env = GridEnvironment(rows, cols)
            self.canvas.config(
                width=self.env.cols * self.cell_size,
                height=self.env.rows * self.cell_size
            )
            self.draw_grid()

    def generate_random_map(self) -> None:
        """Generate a random map with obstacles and rewards."""
        obstacle_prob = simpledialog.askfloat(
            "Random Map",
            "Enter obstacle probability (0-1):",
            parent=self.root,
            minvalue=0.0,
            maxvalue=1.0,
            initialvalue=0.2
        )

        reward_prob = simpledialog.askfloat(
            "Random Map",
            "Enter reward probability (0-1):",
            parent=self.root,
            minvalue=0.0,
            maxvalue=1.0,
            initialvalue=0.1
        )

        if obstacle_prob is not None and reward_prob is not None:
            self.env.generate_random_map(obstacle_prob, reward_prob)
            self.draw_grid()

    def clear_environment(self) -> None:
        """Clear all obstacles and rewards from the environment."""
        self.env.reset_environment()
        self.draw_grid()

    def save_environment(self) -> None:
        """Save the current environment to a JSON file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.env.to_dict(), f, indent=2)
                messagebox.showinfo("Success", "Environment saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save environment: {str(e)}")

    def load_environment(self) -> None:
        """Load an environment from a JSON file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.env = GridEnvironment.from_dict(data)
                self.canvas.config(
                    width=self.env.cols * self.cell_size,
                    height=self.env.rows * self.cell_size
                )
                self.draw_grid()
                messagebox.showinfo("Success", "Environment loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load environment: {str(e)}")


def main():
    """Main function to start the application."""
    root = tk.Tk()
    env = GridEnvironment(10, 10)
    gui = GridEnvironmentGUI(root, env)
    root.mainloop()


if __name__ == "__main__":
    main()