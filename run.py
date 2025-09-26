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
    
    for episode in range(config['training']['episodes']):
        episode_start_time = time.time()
        state = env.reset()
        total_reward, episode_trajectory, ep_loss = 0, [], 0
        
        for step in range(config['training']['max_steps']):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train()
            ep_loss += loss if loss else 0
            state = next_state
            total_reward += reward
            episode_trajectory.append(env.agent_pos)
            
            if env.show_maze:
                env.render()
            
            if done:
                break
        
        episode_time = time.time() - episode_start_time
        logger.log_episode(total_reward, ep_loss, episode_trajectory,
                         episode_time, step + 1, agent.epsilon)
        
        agent.update()
        
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