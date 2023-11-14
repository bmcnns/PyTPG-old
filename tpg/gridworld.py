import numpy as np

class GridWorld:
    def __init__(self, width, height, targetCell, walls):
        self.width = width
        self.height = height
        self.targetCell = targetCell
        self.agentPosition = (0, 0)
        self.walls = walls
        self.rewards = self.generateRewards()

    def isValidPosition(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def generateRewards(self):
        rewards = np.zeros((self.height, self.width))

        # Destination reward
        rewards[self.targetCell[1]][self.targetCell[0]] = 100

        for wall in self.walls:
            rewards[wall[1]][wall[0]] = -1

        return rewards

    def step(self, action):
        dx, dy = 0, 0

        # Move up
        if action == 0:
            dy = -1
        # Move down
        elif action == 1:
            dy = 1
        # Move left
        elif action == 2:
            dx = -1
        # Move right
        elif action == 3:
            dx = 1

        new_x, new_y = self.agentPosition[0] + dx, self.agentPosition[1] + dy

        if self.isValidPosition(new_x, new_y) and (new_x, new_y) not in self.walls:
            self.agentPosition = (new_x, new_y)
            reward = self.rewards[new_y][new_x]
        else:
            # Apply negative reward for hitting a wall
            reward = -1

        return self.getState(), reward

    def getState(self):
        state = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) == self.agentPosition:
                    state.append(1)  # Agent
                elif (x, y) == self.targetCell:
                    state.append(2)  # Target
                elif (x, y) in self.walls:
                    state.append(3)  # Wall
                else:
                    state.append(0)  # Empty cell

        return state

    def isTerminal(self):
        return self.agentPosition == self.targetCell

    def reset(self):
        self.agentPosition = (0, 0)

    def display(self):
        grid = np.zeros((self.height, self.width), dtype=int)
        grid[self.agentPosition[1]][self.agentPosition[0]] = 1  # Agent
        grid[self.targetCell[1]][self.targetCell[0]] = 2  # Target

        for wall in self.walls:
            grid[wall[1]][wall[0]] = 3

        for row in grid:
            print(" ".join(["0" if cell == 0 else "P" if cell == 1 else "G" if cell == 2 else "X" if cell == 3 else "UNKNOWN" for cell in row]))

        print()