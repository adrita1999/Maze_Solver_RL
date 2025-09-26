# Reinforcement Learning Maze Solver

## Introduction

This project implements a **maze-solving** environment using **Reinforcement Learning (RL)**. It allows students to compare **Deep Q-Networks (DQN)** and **Q-Learning** in solving a maze by experimenting with different hyperparameters and epsilon strategies (decay, Boltzmann, performance-based). The project logs results and generates visualizations to analyze performance.

## Installation

We recommend creating a virtual environment to install the required packages. Specifically, we recommend using `anaconda` or `virtualenv`. To create a virtual environment using `anaconda`, run the following command:

```bash
conda create -n project_3 python=3.10
conda activate project_3
```

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

Finally, you will need to install PyTorch. PyTorch installation will depend on your system configuration. To install PyTorch, follow the instructions on the [official website](https://pytorch.org/get-started/locally/).

## Maze Creation

You can create custom mazes using the Maze Editor. To create an N x N maze, run the following command:

```bash
python rl/maze_editor.py --size 10
```

You should then follow the instructions in the terminal to create the maze and save it to a file. This file can then be used in your experiments.

## Experiment Configs 
Experiments are configured using YAML files. Below is the structure of a sample config (DQN_decay.yaml):

```yaml
experiment_name: "DQN_Decay_10x10"
device: cuda  # Use "cpu" if GPU is not available
agent: "DQN"
maze_file: "mazes/10x10.txt"

hyperparameters:
  hidden_size: 256
  gamma: 0.99
  epsilon_policy: "decay"
  epsilon: 1.0
  epsilon_min: 0.1
  epsilon_decay: 0.99
  batch_size: 128
  target_update: 5

training:
  episodes: 500
  max_steps: 200
  render: True

viz:
  num_plots: 4
  window_size: null  # Auto-detect window size

save_dir: "results/dqn_decay/"
```

Modify this file to change parameters such as:
- `maze_file`: Path to the maze file
- `epsilon_policy`: Epsilon strategy (decay, Boltzmann, performance-based)
- `epsilon`: Initial epsilon value ...

It will be up to you to create the necessary YAML files for your experiments and understand the impact what each parameter is doing.

## Saved Results and Visualization

After training, results are saved in the experiment's `save_dir`, e.g., `results/dqn_decay/`. The following files are stored:

- **`config.yaml`** - The YAML configuration used for the experiment
- **`rewards.npy`** - Rewards per episode
- **`epsilons.npy`** - Exploration rate over time
- **`episode_times.npy`** - Time taken per episode
- **`episode_steps.npy`** - Steps taken per episode
- **`trajectories.npy`** - Agent's movements
- **`summary.npy`** - Experiment summary (average reward, steps, etc.)

### Visualizations:
The project generates the following visualizations:
1. **Reward Progression**: Plots episode rewards with a moving average
2. **Steps Per Episode**: Shows the number of steps taken in each episode
3. **Epsilon Decay**: Tracks the exploration rate over episodes
4. **Episode Duration**: Displays time taken per episode
5. **Trajectory Heatmaps**: Highlights frequently visited locations in the maze

All visualizations are saved as images in the `save_dir`.  

### Important Note:

The saved results can and should be used in your analysis and report. The current visualizations are just there to demonstrate how to load and visualize results. It will be up to you to create the necessary visualizations for your analysis.


## Quickstart

To run an experiment, use the following command:

```bash
python run.py --config configs/DQN_decay.yaml
```

#Extra credits:
If you attempt any extra credit describe here.
