import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# Initialization
N = 100
initial_states = np.random.choice([Cell.ON, Cell.OFF], N*N, p=[0.2, 0.8])
grid = np.array([Cell(state) for state in initial_states]).reshape(N, N)

# Set neighbors
for i in range(N):
    for j in range(N):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                grid[i, j].neighbors[di+1][dj+1]=grid[(i+di)%N, (j+dj)%N]

# Animation
fig, ax = plt.subplots()
img = ax.imshow(np.vectorize(lambda x: x.state)(grid), interpolation='nearest')
ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N, ), frames = 50,
                              interval=50, save_count=50)

plt.show()
