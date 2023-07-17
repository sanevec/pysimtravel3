import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Cell:    
	STREET = 255
	FREE = 0

	def __init__(self, initial_state):
		self.state = initial_state
		self.next_state = None
		self.neighbors = [[0,0,0],[0,0,0],[0,0,0]]

	def add_neighbor(self, neighbor):
		self.neighbors.append(neighbor)

	def set_next_state(self):
		total = sum(sum(objeto.state for objeto in fila) for fila in self.neighbors) 
		total -= self.neighbors[1][1].state
		total = total // Cell.STREET
		
		if self.state == Cell.STREET:
			if (total < 2) or (total > 3):
				self.next_state = Cell.FREE
			else:
				self.next_state = self.state
		else:
			if total == 3:
				self.next_state = Cell.STREET
			else:
				self.next_state = self.state

	def update_state(self):
		self.state = self.next_state


def update(frameNum, img, grid, heigh, width):
    for i in range(heigh):
        for j in range(width):
            grid[i, j].set_next_state()
    newGrid = np.empty((heigh, width))
    for i in range(heigh):
        for j in range(width):
            grid[i, j].update_state()
            newGrid[i, j] = grid[i, j].state
    img.set_data(newGrid)
    return img,

class Block:
	def __init__(self):
		self.lanes = [
			[ (0,3), (3,3), (3,-1) ], #rojo
			[ (3,3), (47,3) ], #azul
			[ (3,4), (47,4) ], #azul
			[ (47,2), (3,2) ], #azul
			[ (47,1), (3,1) ], #azul
			[ (4,15), (47,15) ], #amarillo
			[ (47,36), (4,36) ], #amarillo
			[ (15,47), (15,2) ], #amarillo
			[ (36,2), (36,47), ], #amarillo
		]
		max_width = 0
		max_height = 0
		for lane in self.lanes:
			for point in lane:
				if point[0] > max_width:
					max_width = point[0]
				if point[1] > max_height:
					max_height = point[1]
		self.width = max_width*2-1
		self.height = max_height*2-1
	
	def draw(grid,x,y):
		pass 
		

class CityBuilder:
	def __init__(self,verticalBlocks,horizontalBlocks, block):
		self.verticalBlocks = verticalBlocks
		self.horizontalBlocks = horizontalBlocks
		self.grid = np.empty((verticalBlocks,horizontalBlocks))
		
class Grid:
	def __init__(self, heigh, width):
		self.width = width
		self.heigh = heigh
		initial_states = np.random.choice([Cell.FREE,Cell.STREET], width*heigh, p=[0.6,0.4])
		self.grid = np.array([Cell(state) for state in initial_states]).reshape(heigh, width)
		for i in range(heigh):
			for j in range(width):
				for di in [-1, 0, 1]:
					for dj in [-1, 0, 1]:
						self.grid[i, j].neighbors[di+1][dj+1]=self.grid[(i+di)%heigh, (j+dj)%width]

g=Grid(100,100)
# Animation
fig, ax = plt.subplots()
img = ax.imshow(np.vectorize(lambda x: x.state)(g.grid), interpolation='nearest')
ani = animation.FuncAnimation(fig, update, fargs=(img, g.grid, g.heigh, g.width, ), frames = 50,
                              interval=50, save_count=50)

plt.show()
