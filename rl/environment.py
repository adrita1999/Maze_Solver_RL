import numpy as np
import pygame

class MazeEnvironment:
    def __init__(self, size=10, config=None):
        self.size = size
        self.agent_pos = (1, 1)  # Default start position
        self.goal_pos = (size-2, size-2)  # Default goal position
        self.action = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1), "NO_MOVE":(0,0)}
        # Initialize pygame for visualization
        self.show_maze = config['training']['render']
        if self.show_maze:
            pygame.init()
            self.cell_size = 50
            self.screen_size = self.size * self.cell_size
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.agent_image = pygame.image.load('agent.png').convert_alpha()
            self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size, self.cell_size))
            pygame.display.set_caption("Maze Environment")
        
        # Colors
        self.colors = {
            'wall': (0, 0, 0),
            'empty': (255, 255, 255),
            'agent': (255, 0, 0),
            'goal': (0, 255, 0),
            'start': (255,0,0)
        }

        self.extra_rewards={

        }
        
        # Load maze from file or create random maze
        if config:
            self._load_maze(config['maze_file'])
        else:
            self.maze = np.zeros((self.size, self.size))
            # Add walls randomly
            self.maze[np.random.random((self.size, self.size)) < 0.3] = 1
            # Ensure start and goal positions are empty
            self.maze[self.agent_pos] = 0
            self.maze[self.goal_pos] = 0
        
        # Store trajectory for tracking movements
        self.trajectory = []

    def _load_maze(self, maze_file):
        """Load maze configuration from file"""
        with open(maze_file, 'r') as f:
            self.size = int(f.readline().strip())
            self.maze = np.zeros((self.size, self.size))
            for i, line in enumerate(f):
                line = line.strip()
                if i >= self.size:
                    break
                for j, char in enumerate(line[:self.size]):
                    if char == 'S':
                        self.agent_pos = (i, j)
                        self.start_pos = (i, j)
                    elif char == 'G':
                        self.goal_pos = (i, j)
                    elif char == '#':
                        self.maze[i, j] = 1

    def reset(self):
        """Reset environment to initial state"""
        self.agent_pos = self.start_pos if hasattr(self, 'start_pos') else (1, 1)
        self.trajectory = [self.agent_pos]
        return self._get_observation()

    def _get_observation(self):
        """Get values of adjacent cells """
        x, y = self.agent_pos
        obs = []
        for dx, dy in [self.action["UP"], self.action["DOWN"],
                       self.action["RIGHT"], self.action["LEFT"],
                       self.action["NO_MOVE"]]:  # Up, Down, Left, Right
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                obs.append(self.maze[new_x, new_y])
            else:
                obs.append(1)  # Treat out of bounds as walls
        return np.array(obs)

    def step(self, action):
        """Take action in environment"""
        directions = [self.action["UP"], self.action["DOWN"],
                      self.action["RIGHT"], self.action["LEFT"],
                      self.action["NO_MOVE"]]
        dx, dy = directions[action]
        x, y = self.agent_pos
        new_x, new_y = x + dx, y + dy
        
        if (0 <= new_x < self.size and
                0 <= new_y < self.size and
                self.maze[new_x, new_y] != 1):
            self.agent_pos = (new_x, new_y)
            self.trajectory.append(self.agent_pos)
            
        done = self.agent_pos == self.goal_pos
        if done:
            reward = 1
        else:
            reward = -0.01
        return self._get_observation(), reward, done

    def render(self):
        """Render current state of environment"""
        # self.screen.fill((255, 255, 255))
        for i in range(self.size):
            for j in range(self.size):
                color = self.colors['wall'] if self.maze[i,j] == 1 else self.colors['empty']
                pygame.draw.rect(self.screen, color,
                               (j*self.cell_size, i*self.cell_size,
                                self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (200, 200, 200),
                               (j*self.cell_size, i*self.cell_size,
                                self.cell_size, self.cell_size), 1)
        pygame.draw.rect(self.screen, self.colors['start'],
                        (self.start_pos[1]*self.cell_size, self.start_pos[0]*self.cell_size,
                         self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, self.colors['goal'],
                         (self.goal_pos[1] * self.cell_size, self.goal_pos[0] * self.cell_size,
                          self.cell_size, self.cell_size))
        self.screen.blit(self.agent_image,
                         (self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size))
        pygame.display.flip()
    
    def close(self):
        """Close the environment"""
        pygame.quit()
        pass
        
