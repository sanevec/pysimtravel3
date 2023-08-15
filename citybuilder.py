import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import random
import math

class Cell:    
	ONE=1
	TWO=2
	THREE=3
	FREE = 0

	def __init__(self, initial_state):
		self.state = initial_state
		self.next_state = None
		self.neighbors = [[0,0,0],[0,0,0],[0,0,0]]
		self.origin=[]
		self.destination=[]
		self.car=None
		self.velocity=0
		self.x=-1
		self.y=-1
		self.t=0

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

	def update(self,t): 
		# when it is an intersection
		stop=False
		for origin in reversed(self.origin):
			if not stop:
				if origin.car is None:
					continue
				stop=True
			else:
				origin.t=t+1

class Block:
	def __init__(self):
		r = 1
		self.lanes = [
			# [ (-1,3), (3,3), (3,-1) ], #rojo
			[ (r+1,3), (r+1,-1)],
			[ (r,3), (r,-1)],
			[(-1,r+1),(3,r+1)],
			[(-1,r),(3,r)],


			[ (r+1,47), (r+1,3) ], #azul
			[ (r,47), (r,3) ], #azul

			[ (3,r+1), (47,r+1) ], #azul
			[ (3,r), (47,r) ], #azul

			[ (47,15), (r+1,15) ], #amarillo
			[ (r+1,36), (47,36) ], #amarillo			
			[ (15,r+1), (15,47) ], #amarillo
			[ (36,47), (36,r+1), ], #amarillo
		]
		self.velocities = [
			1,
			1,
			1,

			1,
			2,
			2,
			2,
			2,
			1,
			1,
			1,
			1,
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

	def draw2(self,grid,lastx,lasty,xx,yy,velocity):
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
				grid.link(last,current,velocity)
				last=current
				yield
		if lasty == yy:
			inc = -1
			if lastx < xx:
				inc = 1
			for i in range(lastx,xx+inc,inc):
				current=grid.grid[yy%grid.heigh,i%grid.width]
				grid.link(last,current,velocity)
				last=current
				yield
	def draw(self,grid,x,y):
		for i,lane in enumerate(self.lanes):
			lastx=None
			lasty=None
			for point in lane:
				xx=x+point[0]+1
				yy=y+point[1]+1
				#grid.grid[xx][yy].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i]):
					yield
				lastx=xx
				lasty=yy
			lastx=None
			lasty=None
			for point in lane:
				xx=x+point[1]+1
				yy=y-point[0]-1
				#grid.grid[x+point[1]+1,y-point[0]-1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i]):
					yield
				lastx=xx
				lasty=yy
			lastx=None
			lasty=None
			for point in lane:
				xx=x-point[0]-1
				yy=y-point[1]-1
				#grid.grid[x-point[0]-1,y-point[1]-1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i]):
					yield
				lastx=xx
				lasty=yy
			lastx=None
			lasty=None
			for point in lane:
				xx=x-point[1]-1
				yy=y+point[0]+1
				#grid.grid[x-point[1]-1,y+point[0]+1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i]):
					yield
				lastx=xx
				lasty=yy
		
class City:
	def __init__(self,verticalBlocks,horizontalBlocks, block):
		self.verticalBlocks = verticalBlocks
		self.horizontalBlocks = horizontalBlocks
		self.block=block
		self.grid=Grid(verticalBlocks*block.height,horizontalBlocks*block.width)
		self.t=0

	def generator(self):
		yieldCada=1000
		yieldI=0
		for i in range(self.verticalBlocks):
			for j in range(self.horizontalBlocks):
				for _ in self.block.draw(self.grid,i*self.block.height+self.block.height//2,j*self.block.width+self.block.width//2):
					if yieldCada<=yieldI:
						yieldI=0
						yield
					yieldI+=1
		
		self.cars=[]
		for cars in range(500*self.verticalBlocks*self.horizontalBlocks): # number of cars
			self.cars.append(Car(self.grid,self.grid.randomStreet(),self.grid.randomStreet()))
			if yieldCada<=yieldI:
				yieldI=0
				yield
			yieldI+=1

		while True:
			self.t+=1
			moreMove=self.cars
			noMove=[]
			while True:
				moreMove2=[]
				for car in moreMove:
					move=car.move(self.t)
					if move:
						moreMove2.append(car)
					else:
						noMove.append(car)
				if len(moreMove2)==0:
					break
				moreMove=moreMove2

			yieldCada=1
			self.cars=noMove
			for inter in self.grid.intersections:
				inter.update(self.t)
			if yieldCada<=yieldI:
				yieldI=0
				yield
			yieldI+=1

class Grid:
	def __init__(self, heigh, width):
		self.width = width
		self.heigh = heigh
		self.intersections = []
		initial_states = np.random.choice([Cell.FREE], width*heigh, p=[1])
		self.grid = np.array([Cell(state) for state in initial_states]).reshape(heigh, width)
		for i in range(heigh):
			for j in range(width):
				self.grid[i, j].x=j
				self.grid[i, j].y=i
				for di in [-1, 0, 1]:
					for dj in [-1, 0, 1]:
						self.grid[i, j].neighbors[di+1][dj+1]=self.grid[(i+di)%heigh, (j+dj)%width]
	def distance(self,x0,y0,x1,y1):
		# latince distance
		dx = abs(x1 - x0)
		dy = abs(y1 - y0)
		if dx > self.width / 2:
			dx = self.width - dx
		if dy > self.heigh / 2:
			dy = self.heigh - dy
		return dx + dy
	
	def randomStreet(self):
		while True:
			x = random.randint(0,self.width-1)
			y = random.randint(0,self.heigh-1)
			cell=self.grid[y][x]
			if cell.state!=Cell.FREE and cell.car==None:
				return (y,x)
	def link(self,origin,target,velocity):
		if origin!=None:
			target.origin.append(origin)
			origin.destination.append(target)
			target.updateColor()
			origin.updateColor()
			target.velocity=velocity
			origin.velocity=velocity
			if len(target.origin)>1:
				self.intersections.append(target)

class Car:
	def __init__(self, grid, xy,targetxy):
		self.grid = grid
		self.x = xy[0]
		self.y = xy[1]
		self.targetx=targetxy[0]
		self.targety=targetxy[1]
		self.grid.grid[self.y, self.x].car = self
		self.queda=0

	def inTarget(self):
		return self.x == self.targetx and self.y == self.targety

	def move(self,t):
		if self.inTarget():
			(y,x)=self.grid.randomStreet()
			self.targetx=x
			self.targety=y
		
		cell=self.grid.grid[self.y,self.x]
		if t!=cell.t:
			self.queda=cell.velocity
		cell.t=t
		if len(cell.destination)==1:
			toCell=cell.destination[0]
			if toCell.t==t or toCell.car!=None:
				return False #ocupado por otro
			self.x = toCell.x
			self.y = toCell.y
			cell.car = None
			toCell.car = self
			toCell.t=t
		else:
			ire=self.aStart(cell)
			if ire.t==t or ire.car!=None:
				return False
			self.x = ire.x
			self.y = ire.y
			cell.car = None
			ire.car = self
			ire.t=t
	
		self.queda-=1
		if self.queda==0:
			return False
		return True

	def aStart(self,cell):
		# only mark visited if it has more than one destination
		visited=set()
		visited.add(cell)
		opened={}
		for d in cell.destination:
			opened[d]=d
		opened2={}
		while True:
			# solo se añaden los visited con mas de uno
			for (o,r) in opened.items():
				if o.x==self.targetx and o.y==self.targety:
					return opened[o]
				if len(o.destination)==1:
					opened2[o.destination[0]]=r
				else:
					if o not in visited:
						visited.add(o)
						for d in o.destination:
							opened2[d]=r
			opened=opened2
			opened2={}

	
city=City(1,1,Block())
city_generator = city.generator()
g=city.grid
#g=Grid(100,100)

def update(frameNum, img, grid, heigh, width):
	try:
		#for i in range(100):
		next(city_generator)
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
			newGrid[i, j] = cell2color(grid[i, j])
			# if newGrid[i, j] == Cell.STREET:
			# 	print(i,j)
			
	# initial_states = np.random.choice([Cell.STREET, Cell.FREE], grid.shape[0]*grid.shape[1] , p=[0.2, 0.8])
	# newGrid = np.array([state for state in initial_states]).reshape(grid.shape[0], grid.shape[1])
	img.set_data(newGrid)
	return img,

# Animation
next(city_generator)
next(city_generator)

fig, ax = plt.subplots()

bounds = [0, 1, 2, 3, 4, 5]
cmap = colors.ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'white'])
norm = colors.BoundaryNorm(bounds, cmap.N)

def cell2color(cell):
	if cell.state==Cell.FREE:
		return 0
	r=len(cell.destination)+1#+len(cell.origin)
	if r>4:
		print("debug")
	if cell.t==city.t+1:
		r=1
	if cell.car!=None:
		r=4
	return r

img = ax.imshow(np.vectorize(cell2color)(g.grid), interpolation='nearest', cmap=cmap, norm=norm)
ani = animation.FuncAnimation(fig, update, fargs=(img, g.grid, g.heigh,g.width, ), frames=50,interval=1)
plt.show()