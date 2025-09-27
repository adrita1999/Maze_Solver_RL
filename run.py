import argparse
import numpy as np
import yaml
import os
import time
from rl.environment import MazeEnvironment
from rl.DQN_agent import DQNAgent
from rl.Q_learning_agent import QLearningAgent
from rl.logger import ExperimentLogger
from rl.viz import analyze_experiment

# Define parameter filters for different policies
policy_param_keys = {
    "decay": ["epsilon", "epsilon_min", "epsilon_decay"],
    "boltzmann": ["temperature", "temperature_min", "temperature_decay"],
    "performance_based": ["epsilon", "epsilon_min", "epsilon_decay"],
}

def load_experiment_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train(config):
    # Create timestamped experiment directory
    timestamp = time.strftime('%Y%m%d')#_%H%M%S')
    save_dir = os.path.join(config['save_dir'], f"{config['experiment_name']}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config for reproducibility
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    env = MazeEnvironment(size=10, config=config)
    
    # Define parameter filters for different policies
    policy_param_keys = {
        "decay": ["epsilon", "epsilon_min", "epsilon_decay"],
        "boltzmann": ["temperature", "temperature_min", "temperature_decay"],
        "performance_based": ["epsilon", "epsilon_min", "epsilon_decay"],
    }

    # Select relevant parameters based on epsilon policy
    policy = config['hyperparameters']['epsilon_policy']
    policy_params = {k: v for k, v in config['hyperparameters'].items() if k in policy_param_keys[policy]}

    if config['agent'] == 'Q-learning':
        agent = QLearningAgent(
            state_size=5, action_size=5,
            alpha=config['hyperparameters']['alpha'],
            gamma=config['hyperparameters']['gamma'],
            epsilon_policy=policy,
            **policy_params  # Dynamically unpack relevant params
        )
    else:
        agent = DQNAgent(
            state_size=5, action_size=5,
            num_layers=config['hyperparameters']['num_layers'],
            hidden_size=config['hyperparameters']['hidden_size'],
            gamma=config['hyperparameters']['gamma'],
            batch_size=config['hyperparameters']['batch_size'],
            target_update=config['hyperparameters']['target_update'],
            device=config.get('device', 'cpu'),
            epsilon_policy=policy,
            **policy_params  # Dynamically unpack relevant params
        )

    
    logger = ExperimentLogger(save_dir)
    

    # Loop over the total number of training episodes
    for episode in range(config['training']['episodes']):
        episode_start_time = time.time() # Record the start time of this episode for timing stats
        # Reset the environment to the starting position
        state = env.reset()
        # Initialize counters for total reward, trajectory, and loss in this episode
        total_reward, episode_trajectory, ep_loss = 0, [], 0
        
        # Limit the number of steps per episode to prevent infinite loops
        for step in range(config['training']['max_steps']):
            # Agent chooses an action based on the current state
            action = agent.act(state)
            # Environment executes the action: returns next state, reward, and whether goal is reached
            next_state, reward, done = env.step(action)
            # Update the Q-table with this transition
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train() 
            ep_loss += loss if loss else 0 # Track cumulative loss
            state = next_state # Move to the next state
            total_reward += reward # Accumulate total reward earned in this episode
            episode_trajectory.append(env.agent_pos) # Log agent's position after this move
            
            # If rendering is enabled, visually display the maze after each step
            if env.show_maze:
                env.render()
            
            # If the agent reaches the goal, exit the loop early (episode complete)
            if done:
                break
        
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, "
              f"Time={episode_time:.2f}s",end=' '
              )
        if agent.epsilon:
            print(f"Epsilon={agent.epsilon:.2f}, ",end=' ')
        print()

    env.close()
    logger.save_logs()
    
    # Generate visualizations
    maze = env.maze  # Get the maze layout for visualization
    
    # Viz params
    num_episodes = config['training']['episodes']
    num_plots = config['viz']['num_plots']
    if num_episodes < num_plots:
        episodes_to_plot = np.arange(num_episodes)
    else:
        # Calculate indices for middle episodes
        middle_episodes = np.linspace(0, num_episodes - 1, num_plots, dtype=int)
        episodes_to_plot = np.unique(middle_episodes)  # Remove any duplicates
    
    window = max(5, min(50, num_episodes // 10)) if config['viz']['window_size'] is None else config['viz']['window_size'] 
    analyze_experiment(save_dir, maze, episodes_to_plot, window)

    
    return save_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default="configs/Q_learning_decay.yaml",help='configs/Q_learning_decay.yaml')
    args = parser.parse_args()
    
    config = load_experiment_config(args.config)
    save_dir = train(config)
    print(f"\nExperiment completed. Results saved to: {save_dir}")

if __name__ == "__main__":
    main()