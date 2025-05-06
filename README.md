# Grid Pathfinding with Q-Learning and Deep Q-Network

This repository contains implementations of classic Q-Learning and Deep Q-Network (DQN) agents for solving grid-based pathfinding tasks using a customizable environment.


## 🚀 Features

* **Customizable Grid Environment**: Define obstacles, rewards, start, and goal positions via JSON or GUI.
* **Q-Learning Agent**: Tabular Q-Learning with epsilon-greedy exploration and policy visualization.
* **Deep Q-Network (DQN)**: Convolutional network processing grid observations with experience replay and target network updates.
* **Visualization**: Real-time rendering of training progress, policy arrows, and performance metrics.

## 📦 Installation
 Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### 1. Q-Learning Agent

```bash
python q_learning.py
```

* Adjust parameters such as `training_episodes`, `learning_rate`, and `epsilon` inside `q_learning.py`.
* The GUI will render grid, update episode info, and display the learned policy with arrows.

### 2. Deep Q-Network Agent

```bash
python deep_q_learning.py
```

* Modify training settings (`TRAIN_MAPS`, `training_episodes`, `learning_rate`) at the top of the script.
* After training, trained weights are saved as `dqn_policy.pth` (uncomment save/load lines).
* Evaluation prints average reward and success rate.

### 3.  Map Editor 

```bash
python grid.py
```

* Edit obstacles, rewards, start/goal positions via GUI.
* Save/load custom maps in JSON format for training.

## ⚙️ Configuration

* Map files (`map*.json`) define grid size, cell obstacles, rewards, start, and goal positions.
* Update `GridEnv.load_map()` calls to point to your custom map files.



## 📄 Sample Map JSON

```json
{
  "rows": 10,
  "cols": 10,
  "grid": [
    [{ "obstacle": false, "reward": 0.0 }, ... ],
    ...
  ],
  "start_pos": [0, 0],
  "goal_pos": [9, 9]
}
```

