import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from .model import DQN

class DQNAgent:
    def __init__(self, state_size=5, action_size=5, hidden_size=128, gamma=0.99, batch_size=128,num_layers=2,
                 target_update=5, device="cpu", epsilon_policy="decay", **kwargs):
        self.device = torch.device("cuda" if device == 'cuda' and torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon_policy = epsilon_policy
        self.num_layers = num_layers

        # Assign only relevant params dynamically
        if epsilon_policy == "decay":
            self.epsilon = kwargs.get("epsilon", 1.0)
            self.epsilon_min = kwargs.get("epsilon_min", 0.01)
            self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        elif epsilon_policy == "boltzmann":
            self.epsilon = None
            self.temperature = kwargs.get("temperature", 1.0)
            self.temp_min = kwargs.get("temperature_min", 0.1)
            self.temp_decay = kwargs.get("temperature_decay", 0.995)
        elif epsilon_policy == "performance_based":
            self.epsilon = kwargs.get("epsilon", 1.0)
            self.epsilon_min = kwargs.get("epsilon_min", 0.01)
            self.epsilon_decay = kwargs.get("epsilon_decay", 0.99)

        # Initialize networks
        self.policy_net = DQN(input_size=state_size, hidden_size=hidden_size, output_size=action_size,num_layers=num_layers).to(self.device)
        self.target_net = DQN(input_size=state_size, hidden_size=hidden_size, output_size=action_size,num_layers=num_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)
        self.recent_rewards = deque(maxlen=100)


    def act(self, state):
        if self.epsilon_policy == "boltzmann":
            return self.boltzmann_action(state)

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state).argmax().item()

    def boltzmann_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state).cpu().numpy().flatten()
            # Use temperature parameter for controlling exploration
            exp_q = np.exp(q_values / max(self.temperature, 1e-3))
            probabilities = exp_q / np.sum(exp_q)
            return np.random.choice(self.action_size, p=probabilities)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.recent_rewards.append(reward)

    def get_recent_rewards(self, n):
        """Helper function to safely get n most recent rewards"""
        rewards_list = list(self.recent_rewards)
        return rewards_list[-n:] if len(rewards_list) >= n else rewards_list

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon_policy == "decay":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        elif self.epsilon_policy == "performance_based":
            # Get recent rewards safely using the helper function
            recent_rewards = self.get_recent_rewards(20)  # Get last 20 rewards
            if len(recent_rewards) >= 20:  # Only update if we have enough history
                recent_mean = np.mean(recent_rewards[-10:])  # Last 10 rewards
                old_mean = np.mean(recent_rewards[:10])      # Previous 10 rewards
                if recent_mean > old_mean:
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        elif self.epsilon_policy == "boltzmann":
            # Update temperature (analogous to epsilon decay)
            self.temperature = max(self.temp_min, self.temperature * self.temp_decay)