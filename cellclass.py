import numpy as np

class Cell:    
	ON = 255
	OFF = 0

	def __init__(self, initial_state):
		self.state = initial_state
		self.next_state = None
		self.neighbors = [[0,0,0],[0,0,0],[0,0,0]]

	def add_neighbor(self, neighbor):
		self.neighbors.append(neighbor)

	def set_next_state(self):
		total = sum(sum(objeto.state for objeto in fila) for fila in self.neighbors) 
		total -= self.neighbors[1][1].state
		total = total // Cell.ON
		
		if self.state == Cell.ON:
			if (total < 2) or (total > 3):
				self.next_state = Cell.OFF
			else:
				self.next_state = self.state
		else:
			if total == 3:
				self.next_state = Cell.ON
			else:
				self.next_state = self.state

	def update_state(self):
		self.state = self.next_state


def update(frameNum, img, grid, N):
    for i in range(N):
        for j in range(N):
            grid[i, j].set_next_state()
    newGrid = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            grid[i, j].update_state()
            newGrid[i, j] = grid[i, j].state
    img.set_data(newGrid)
    return img,