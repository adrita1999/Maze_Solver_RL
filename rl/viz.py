import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

def get_safe_window_size(data_length):
    """Determine a safe window size based on data length."""
    if data_length < 3:
        return 1
    elif data_length < 10:
        return 2
    else:
        # Window size should be smaller than data length and reasonable for smoothing
        return min(50, max(3, data_length // 10))

def moving_average(data, window):
    """Compute moving average with proper padding."""
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')

def compute_confidence_interval(data, window):
    """Safely compute confidence intervals for the moving average."""
    if len(data) < window:
        return np.zeros(1), np.zeros(1)
    
    # Pad the data to handle the edges
    pad_data = np.pad(data, (window//2, window//2), mode='edge')
    rolling_windows = np.array([pad_data[i:i+window] for i in range(len(data))])
    std_err = stats.sem(rolling_windows, axis=1)
    return 1.96 * std_err

def plot_training_metrics(save_dir, window=None):
    """Plots comprehensive training metrics including rewards, epsilon, and time statistics."""
    # Load data
    rewards = np.load(os.path.join(save_dir, 'rewards.npy'))
    epsilons = np.load(os.path.join(save_dir, 'epsilons.npy'))
    episode_times = np.load(os.path.join(save_dir, 'episode_times.npy'))
    episode_steps = np.load(os.path.join(save_dir, 'episode_steps.npy'))

    np.savetxt(os.path.join(save_dir, 'rewards.csv'), rewards, delimiter=',')
    np.savetxt(os.path.join(save_dir, 'epsilons.csv'), epsilons, delimiter=',')
    np.savetxt(os.path.join(save_dir, 'episode_times.csv'), episode_times, delimiter=',')
    np.savetxt(os.path.join(save_dir, 'episode_steps.csv'), episode_steps, delimiter=',')

    # Determine safe window size if not provided
    if window is None or window >= len(rewards):
        window = get_safe_window_size(len(rewards))
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))
    episodes = np.arange(len(rewards))
    
    # Plot rewards
    if len(rewards) > 0:
        axes[0].plot(episodes, rewards, alpha=0.3, label='Raw Rewards', color='blue')
        if len(rewards) >= window:
            smoothed_rewards = moving_average(rewards, window)
            smooth_episodes = episodes[window-1:]
            confidence_interval = compute_confidence_interval(rewards, window)
            
            axes[0].plot(smooth_episodes, smoothed_rewards, 
                        label=f'Moving Avg ({window})', color='red')
            axes[0].fill_between(smooth_episodes, 
                               smoothed_rewards - confidence_interval[:len(smoothed_rewards)],
                               smoothed_rewards + confidence_interval[:len(smoothed_rewards)],
                               alpha=0.2, color='red')
                        # --------------------------
            #   Convergence Detection
            # --------------------------
            # We'll look for the first point where the difference in smoothed rewards
            # is below 'convergence_threshold' for 'patience' consecutive episodes.
            
            consecutive_count = 0
            convergence_threshold = 0.01
            patience = 15
            last_value = smoothed_rewards[0]
            convergence_episode = None
            
            for i in range(1, len(smoothed_rewards)):
                diff = abs(smoothed_rewards[i] - last_value)
                if diff < convergence_threshold:
                    consecutive_count += 1
                else:
                    consecutive_count = 0  # reset
                
                last_value = smoothed_rewards[i]
                
                # Once we see 'patience' consecutive episodes with small change, mark convergence
                if consecutive_count >= patience:
                    # Convert index in smoothed array back to original episode number
                    # The i-th entry in smoothed_rewards corresponds to 'smooth_episodes[i]'
                    convergence_episode = smooth_episodes[i]
                    break
            
            if convergence_episode is not None:
                print(f"Detected approximate convergence at episode {convergence_episode}")
            else:
                print("No clear convergence detected based on this heuristic.")
    
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Reward Progression")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot steps per episode
    if len(episode_steps) > 0:
        axes[1].plot(episodes, episode_steps, alpha=0.3, label='Raw Steps', color='blue')
        if len(episode_steps) >= window:
            smoothed_steps = moving_average(episode_steps, window)
            smooth_episodes = episodes[window-1:]
            axes[1].plot(smooth_episodes, smoothed_steps, 
                        label=f'Moving Avg ({window})', color='red')
    
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Steps per Episode")
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot epsilon progression
    if len(epsilons):
        axes[2].plot(episodes, epsilons, label='Epsilon', color='green')
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Epsilon")
        axes[2].set_title("Exploration Rate")
        axes[2].grid(True)
    
    # Plot episode duration
    if len(episode_times) > 0:
        axes[3].plot(episodes, episode_times, alpha=0.3, label='Raw Times', color='blue')
        if len(episode_times) >= window:
            smoothed_times = moving_average(episode_times, window)
            smooth_episodes = episodes[window-1:]
            axes[3].plot(smooth_episodes, smoothed_times, 
                        label=f'Moving Avg ({window})', color='red')
    
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Time (seconds)")
    axes[3].set_title("Episode Duration")
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

    # # ----------------------------------
    # # Separate Plots for Each Metric
    # # ----------------------------------
    # # Rewards Plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(episodes, rewards, alpha=0.3, label='Raw Rewards', color='blue')
    # if len(rewards) >= window:
    #     smoothed_rewards = moving_average(rewards, window)
    #     smooth_episodes = episodes[window-1:]
    #     confidence_interval = compute_confidence_interval(rewards, window)
    #     plt.plot(smooth_episodes, smoothed_rewards, label=f'Moving Avg ({window})', color='red')
    #     plt.fill_between(smooth_episodes, 
    #                      smoothed_rewards - confidence_interval[:len(smoothed_rewards)],
    #                      smoothed_rewards + confidence_interval[:len(smoothed_rewards)],
    #                      alpha=0.2, color='red')
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("Reward Progression")
    # plt.legend()
    # plt.grid(True)
    # rewards_fig_path = os.path.join(save_dir, 'rewards_metrics.png')
    # plt.savefig(rewards_fig_path)
    # plt.close()
    # # print(f"Rewards metrics figure saved to {rewards_fig_path}")
    
    # # Steps Plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(episodes, episode_steps, alpha=0.3, label='Raw Steps', color='blue')
    # if len(episode_steps) >= window:
    #     smoothed_steps = moving_average(episode_steps, window)
    #     smooth_episodes = episodes[window-1:]
    #     plt.plot(smooth_episodes, smoothed_steps, label=f'Moving Avg ({window})', color='red')
    # plt.xlabel("Episode")
    # plt.ylabel("Steps")
    # plt.title("Steps per Episode")
    # plt.legend()
    # plt.grid(True)
    # steps_fig_path = os.path.join(save_dir, 'steps_metrics.png')
    # plt.savefig(steps_fig_path)
    # plt.close()
    # # print(f"Steps metrics figure saved to {steps_fig_path}")
    
    # # Epsilon Plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(episodes, epsilons, label='Epsilon', color='green')
    # plt.xlabel("Episode")
    # plt.ylabel("Epsilon")
    # plt.title("Exploration Rate")
    # plt.legend()
    # plt.grid(True)
    # epsilon_fig_path = os.path.join(save_dir, 'epsilon_metrics.png')
    # plt.savefig(epsilon_fig_path)
    # plt.close()
    # # print(f"Epsilon metrics figure saved to {epsilon_fig_path}")
    
    # # Episode Duration Plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(episodes, episode_times, alpha=0.3, label='Raw Times', color='blue')
    # if len(episode_times) >= window:
    #     smoothed_times = moving_average(episode_times, window)
    #     smooth_episodes = episodes[window-1:]
    #     plt.plot(smooth_episodes, smoothed_times, label=f'Moving Avg ({window})', color='red')
    # plt.xlabel("Episode")
    # plt.ylabel("Time (seconds)")
    # plt.title("Episode Duration")
    # plt.legend()
    # plt.grid(True)
    # times_fig_path = os.path.join(save_dir, 'episode_duration_metrics.png')
    # plt.savefig(times_fig_path)
    # plt.close()
    # # print(f"Episode duration metrics figure saved to {times_fig_path}")

def plot_trajectory_heatmaps(save_dir, maze, episodes=None):
    """Plots trajectory heatmaps for specified episodes or default key points."""
    trajectories = np.load(os.path.join(save_dir, 'trajectories.npy'), allow_pickle=True)
    
    if len(trajectories) == 0:
        print("No trajectory data available")
        return
    
    # Safely determine episodes to plot
    if episodes is None:
        num_plots = min(5, len(trajectories))
        if num_plots == 1:
            episodes = [0]
        else:
            step = (len(trajectories) - 1) // (num_plots - 1)
            episodes = [i * step for i in range(num_plots)]
    else:
        episodes = [ep for ep in episodes if ep < len(trajectories)]
    
    if not episodes:
        print("No valid episodes to plot")
        return
    
    fig, axes = plt.subplots(1, len(episodes), figsize=(4*len(episodes), 4))
    if len(episodes) == 1:
        axes = [axes]
    
    for idx, episode in enumerate(episodes):
        heatmap = np.zeros_like(maze, dtype=float)
        trajectory = trajectories[episode]
        for x, y in trajectory:
            heatmap[x, y] += 1
        
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        axes[idx].imshow(maze, cmap='binary')
        img = axes[idx].imshow(heatmap, cmap='hot', alpha=0.6)
        plt.colorbar(img, ax=axes[idx], label="Visit Frequency")
        axes[idx].set_title(f'Episode {episode + 1}')
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory_heatmaps.png'))
    plt.close()

    # ------------------------------
    # 2. Separate Plot for Each Episode
    # ------------------------------
    for episode in episodes:
        # Create the heatmap for this episode
        heatmap = np.zeros_like(maze, dtype=float)
        trajectory = trajectories[episode]
        for x, y in trajectory:
            heatmap[x, y] += 1
        
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        plt.figure(figsize=(4, 4))
        plt.imshow(maze, cmap='binary')
        img = plt.imshow(heatmap, cmap='hot', alpha=0.6)
        plt.colorbar(img, label="Visit Frequency")
        plt.title(f'Episode {episode + 1}')
        plt.grid(True)
        plt.tight_layout()
        individual_path = os.path.join(save_dir, f'trajectory_heatmap_episode_{episode + 1}.png')
        plt.savefig(individual_path)
        plt.close()

def analyze_experiment(save_dir, maze, episodes_to_plot=None, window=None):
    """Comprehensive analysis of experiment results."""
    # Load summary statistics
    summary = np.load(os.path.join(save_dir, 'summary.npy'), allow_pickle=True).item()
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"Total training time: {summary['total_time']:.2f} seconds")
    print(f"Average episode time: {summary['avg_episode_time']:.2f} seconds")
    print(f"Average episode steps: {summary['avg_episode_steps']:.2f}")
    print(f"Average reward: {summary['avg_reward']:.2f} (±{summary['std_reward']:.2f})")
    print(f"Best reward: {summary['max_reward']:.2f}")
    print(f"Worst reward: {summary['min_reward']:.2f}")
    
    # Generate plots
    plot_training_metrics(save_dir, window)
    plot_trajectory_heatmaps(save_dir, maze, episodes_to_plot)