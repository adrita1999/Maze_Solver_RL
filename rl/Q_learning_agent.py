import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon_policy="decay", **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_policy = epsilon_policy
        self.epsilon = None

        # Assign only relevant params dynamically
        if epsilon_policy == "decay":
            self.epsilon = kwargs.get("epsilon", 1.0)
            self.epsilon_min = kwargs.get("epsilon_min", 0.01)
            self.epsilon_decay = kwargs.get("epsilon_decay", 0.99)
        elif epsilon_policy == "boltzmann":
            self.temperature = kwargs.get("temperature", 1.0)
            self.temp_min = kwargs.get("temperature_min", 0.1)
            self.temp_decay = kwargs.get("temperature_decay", 0.995)
        elif epsilon_policy == "performance_based":
            self.epsilon = kwargs.get("epsilon", 1.0)
            self.epsilon_min = kwargs.get("epsilon_min", 0.01)
            self.epsilon_decay = kwargs.get("epsilon_decay", 0.99)

        # Initialize Q-table
        self.q_table = {}
        self.recent_rewards = []


    def get_state_key(self, state):
        """Convert a continuous or complex state representation into a hashable key."""
        return tuple(np.round(state, decimals=2))

    def act(self, state):
        state_key = self.get_state_key(state)
        if self.epsilon_policy == "boltzmann":
            return self.boltzmann_action(state_key)
        
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        
        # Ensure state exists in Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])  # Exploit
    
    def train(self):
        """ We don't need this method for Q-learning, but it's included for compatibility with our DQN agent. """
        pass

    def boltzmann_action(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        q_values = self.q_table[state_key]
        exp_q = np.exp(q_values / max(self.temperature, 1e-3))
        probabilities = exp_q / np.sum(exp_q)
        return np.random.choice(self.action_size, p=probabilities)
    
    def remember(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        best_next_action = np.argmax(self.q_table[next_state_key])
        target = reward + (1 - done) * self.gamma * self.q_table[next_state_key][best_next_action]
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action])
        
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 20:
            self.recent_rewards.pop(0)
    
    def update(self):
        if self.epsilon_policy == "decay":
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)
        
        elif self.epsilon_policy == "performance_based":
            if len(self.recent_rewards) >= 20:
                recent_mean = np.mean(self.recent_rewards[-10:])
                old_mean = np.mean(self.recent_rewards[:10])
                if recent_mean > old_mean:
                    self.epsilon = max(self.epsilon_min,
                                       self.epsilon * self.epsilon_decay)
        
        elif self.epsilon_policy == "boltzmann":
            self.temperature = max(self.temp_min,
                                   self.temperature * self.temp_decay)