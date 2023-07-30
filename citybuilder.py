import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors

class Cell:    
	ONE=1
	TWO=2
	THREE=3
	STREET = 4
	FREE = 0

	def __init__(self, initial_state):
		self.state = initial_state
		self.next_state = None
		self.neighbors = [[0,0,0],[0,0,0],[0,0,0]]
		self.origin=[]
		self.destination=[]

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

	def link(self,origin):
		if origin!=None:
			self.origin.append(origin)
			origin.destination.append(self)
			self.updateColor()
			origin.updateColor()
	
	def updateColor(self):
		count=len(self.origin)+len(self.destination)
		if count==0:
			self.state=Cell.FREE
		elif count==1:
			self.state=Cell.ONE
		elif count==2:
			self.state=Cell.TWO
		elif count==3:
			self.state=Cell.THREE
		else:
			self.state=Cell.STREET

class Block:
	def __init__(self):
		self.lanes = [
			[ (-1,3), (3,3), (3,-1) ], #rojo
			[ (2,47), (2,3) ], #azul
			[ (1,47), (1,3) ], #azul

			[ (3,2), (47,2) ], #azul
			[ (3,1), (47,1) ], #azul

			[ (47,15), (2,15) ], #amarillo
			[ (2,36), (47,36) ], #amarillo			
			[ (15,2), (15,47) ], #amarillo
			[ (36,47), (36,2), ], #amarillo
		]
		max_width = 0
		max_height = 0
		for lane in self.lanes:
			for point in lane:
				if point[0] > max_width:
					max_width = point[0]
				if point[1] > max_height:
					max_height = point[1]
		self.width = (max_width+1)*2
		self.height = (max_height+1)*2

	def draw2(self,grid,lastx,lasty,xx,yy):
		if lastx is None:
			return
		if lasty is None:
			return
		last=None
		if lastx == xx:
			inc = -1
			if lasty < yy:
				inc = 1
			for i in range(lasty,yy+inc,inc):
				current=grid.grid[i%grid.heigh,xx%grid.width]
				current.link(last)
				last=current
				yield
		if lasty == yy:
			inc = -1
			if lastx < xx:
				inc = 1
			for i in range(lastx,xx+inc,inc):
				current=grid.grid[yy%grid.heigh,i%grid.width]
				current.link(last)
				last=current
				yield
	def draw(self,grid,x,y):
		for lane in self.lanes:
			lastx=None
			lasty=None
			for point in lane:
				xx=x+point[0]+1
				yy=y+point[1]+1
				#grid.grid[xx][yy].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy):
					yield
				lastx=xx
				lasty=yy
			lastx=None
			lasty=None
			for point in lane:
				xx=x+point[1]+1
				yy=y-point[0]-1
				#grid.grid[x+point[1]+1,y-point[0]-1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy):
					yield
				lastx=xx
				lasty=yy
			lastx=None
			lasty=None
			for point in lane:
				xx=x-point[0]-1
				yy=y-point[1]-1
				#grid.grid[x-point[0]-1,y-point[1]-1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy):
					yield
				lastx=xx
				lasty=yy
			lastx=None
			lasty=None
			for point in lane:
				xx=x-point[1]-1
				yy=y+point[0]+1
				#grid.grid[x-point[1]-1,y+point[0]+1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy):
					yield
				lastx=xx
				lasty=yy
		
class CityBuilder:
	def __init__(self,verticalBlocks,horizontalBlocks, block):
		self.verticalBlocks = verticalBlocks
		self.horizontalBlocks = horizontalBlocks
		self.block=block
		self.grid=Grid(verticalBlocks*block.height,horizontalBlocks*block.width)

	def generator(self):
		for i in range(self.verticalBlocks):
			for j in range(self.horizontalBlocks):
				for _ in self.block.draw(self.grid,i*self.block.height+self.block.height//2,j*self.block.width+self.block.width//2):
					yield

class Grid:
	def __init__(self, heigh, width):
		self.width = width
		self.heigh = heigh
		initial_states = np.random.choice([Cell.FREE], width*heigh, p=[1])
		self.grid = np.array([Cell(state) for state in initial_states]).reshape(heigh, width)
		for i in range(heigh):
			for j in range(width):
				for di in [-1, 0, 1]:
					for dj in [-1, 0, 1]:
						self.grid[i, j].neighbors[di+1][dj+1]=self.grid[(i+di)%heigh, (j+dj)%width]


city_builder=CityBuilder(1,1,Block())
city_builder_generator = city_builder.generator()
g=city_builder.grid
#g=Grid(100,100)

def update(frameNum, img, grid, heigh, width):
	try:
		next(city_builder_generator)
	except StopIteration:
		pass
	except Exception as e:
		print(e)
		pass

	# for i in range(heigh):
	#     for j in range(width):
	#         grid[i, j].set_next_state()

	newGrid = np.empty((heigh, width))
	for i in range(heigh):
		for j in range(width):
			#grid[i, j].update_state()
			newGrid[i, j] = grid[i, j].state
			# if newGrid[i, j] == Cell.STREET:
			# 	print(i,j)
			
	# initial_states = np.random.choice([Cell.STREET, Cell.FREE], grid.shape[0]*grid.shape[1] , p=[0.2, 0.8])
	# newGrid = np.array([state for state in initial_states]).reshape(grid.shape[0], grid.shape[1])
	img.set_data(newGrid)
	return img,

# Animation
next(city_builder_generator)
next(city_builder_generator)

fig, ax = plt.subplots()

bounds = [0, 1, 2, 3, 4, 5]
cmap = colors.ListedColormap(['black', 'red', 'green', 'blue', 'yellow'])
norm = colors.BoundaryNorm(bounds, cmap.N)

img = ax.imshow(np.vectorize(lambda x: x.state)(g.grid), interpolation='nearest', cmap=cmap, norm=norm)
ani = animation.FuncAnimation(fig, update, fargs=(img, g.grid, g.heigh,g.width, ), frames=50,interval=1)
plt.show()