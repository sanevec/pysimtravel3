from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import random
from collections import namedtuple
import traceback
import math
from enum import IntEnum
import json
import time

random.seed(0) # To reproduce the same simulation, comment this line to get a random simulation

class Parameters:
	"""
	Parameters of the simulation

	Attributes:
		verticalBlocks (int): Number of vertical blocks
		horizontalBlocks (int): Number of horizontal blocks
		numberCarsPerBlock (int): Number of cars per block
		numberStations (int): Number of charging stations
		numberChargingPerStation (int): Number of charging per station
		carMovesFullDeposity (int): Number of moves when the car is full
		carRechargePerTic (int): Number of moves that the car recharge per tic
		opmitimizeCSSearch (int): Number of charging stations to store in bifurcation cell to optimize the search
		viewDrawCity (bool): If true, draw the city
	"""
	def __init__(self):
		self.verticalBlocks=1
		self.horizontalBlocks=1
		self.numberStationsPerBlock=4# tipical 1/(vb*hb),1, 4 
		self.numberChargersPerBlock=8

		self.numberBlocks=self.verticalBlocks*self.horizontalBlocks
		self.numberStations=self.numberStationsPerBlock*self.numberBlocks
		self.numberChargers=self.numberChargersPerBlock*self.numberBlocks
		self.numberChargingPerStation=self.numberChargers//self.numberStations
		self.numberCars=10
		# to allow to move all cars at medium velocity
		self.mediumVelocity=2
		self.carRechargePerTic=self.numberCars*self.mediumVelocity/self.numberChargers


		self.carMovesFullDeposity=27000


		self.opmitimizeCSSearch=3 # bigger is more slow
		self.carSizeTarget=20

		self.aStarMethod="Time" # Time or Distance
		self.aStarAddCSTimeSteps=False

		# when aStarMethod is Time
		self.aStarAddRoadCarAsTimeSteps=0
		self.aStarUseCellAverageVelocity=True
		self.aStarUseCellExponentialWeight=0.5 # 0 disable, 0-1 weight of old velocity data
		self.aStarCalculateEach=5 # The aStar calculation is slow, so we can calculate it each n bifurcation cells
		self.aStarDeep=100 # bigger is more slow, more precision
		self.aStarRemainderWeight=2 #math.sqrt(2) # weight of lineal distance to target to time
		# self.aStarStepsPerCar=100000 # bigger is more slow, more precision

		# interface parameters
		self.viewDrawCity=False
		self.statsFileName="stats1" # if "" then not save / stats1

class CarPriority(IntEnum):
	'''
	CarPriority is used to define the priority of the execution in the grid.
	Our Cellular Automata is asynchronous. Some cells (with cars) are executed before than others.
	'''
	StopedNoPriority = -2
	StopedPriority = -1
	NoAsigned=0
	Priority = 1
	NoPriority = 2

class CarState(IntEnum):
	'''
	CarState is used to evaluate the efficiency of the ubication of the charging stations.
	'''
	Destination = 0
	Waiting = 1
	Driving = 2
	ToCharging = 3
	ChargingQueue = 4
	ChargingSlot = 5

class ChargingStation:
	"""
	The distance and fuel consumption in this version are the same. Astar will have to adapt when this simplification is changed.
	Charging station is a cell that can charge cars. It has a queue of cars and a number of charging slots.
	The chargins statation (CS) alson has a route map to all cells of the city. This route map is used to calculate the distance to the CS.

	Attributes:
		p (Parameters): Parameters of the simulation
		grid (Grid): Grid of the city
		cell (Cell): Cell of the grid where the CS is located
		numberCharging (int): Number of charging slots
		queue (List[Car]): Queue of cars
		car (List[Car]): List of cars in the charging slots
	"""
	def __init__(self,p,grid,cell):
		self.p=p
		self.grid=grid
		self.cell=cell
		cell.cs=self
		# Note: factorizable
		numberCharging=int(p.numberChargingPerStation)
		self.numberCharging=numberCharging
		self.queue=[]
		self.car=[None for i in range(numberCharging)]
		self.carsInCS=0

		#self.insertInRouteMap(cell)

	def TicksStepsToAttend(self):
		total=0
		for car in self.queue:
			total+=(car.p.carMovesFullDeposity-car.moves)/car.p.carRechargePerTic
		for car in self.car:
			if car!=None:
				total+=(car.p.carMovesFullDeposity-car.moves)/car.p.carRechargePerTic
		return total/self.numberCharging	
	def moveCS(self,t):
		if self.carsInCS==0:
			return
		# If there is a car in the queue, and there is a gap in the station, then the car enters the station
		while 0<len(self.queue) and None in self.car:
			car=self.queue.pop(0)
			i=self.car.index(None)
			self.car[i]=car
			car.state=CarState.ChargingSlot

		# Recharge the cars that are in the charger
		candidateToLeave=-1
		for i in range(self.numberCharging):
			if self.car[i]!=None:
				self.car[i].moves+=self.p.carRechargePerTic
				#print("Recharge percent:",self.car[i].moves/self.p.carMovesFullDeposity*100,"%")
				if self.car[i].moves>self.p.carMovesFullDeposity:
					self.car[i].moves=self.p.carMovesFullDeposity
					candidateToLeave=i

		#candidateToLeave=-1
		if 0<=candidateToLeave and self.cell.car==None:
			car=self.car[candidateToLeave]
			self.car[candidateToLeave]=None
			car.state=CarState.Driving
			car.target2=None
			self.cell.car=car
			self.carsInCS-=1
			car.cell=self.cell
			car.calculateRoute(car.cell,t)

	def insertInRouteMap(self):
		'''
		Insert the CS in the route map of the city.
		This a reverse version of the A* algorithm. 
		It is used to calculate the distance to the CSs near on the bifurcation cells. 
		'''
		cell=self.cell
		visited = []
		distance = 0
		current_level = [cell]

		while current_level:
			next_level = []

			for current_cell in current_level:
				visited.append(current_cell)

				allowInsert=False
				if len(current_cell.destination) > 1:
					# sort h2cs
					# if worst is better than the current distance break propagation
					if len(current_cell.h2cs)+1<self.p.opmitimizeCSSearch:
						allowInsert=True
					else:
						if distance<current_cell.h2cs[-1].distance:
							allowInsert=True
					if allowInsert:
						current_cell.h2cs.append(HeuristicToCS(self, distance))
						current_cell.h2cs.sort(key=lambda x: x.distance)
						current_cell.h2cs=current_cell.h2cs[:self.p.opmitimizeCSSearch]
				else:
					allowInsert=True

				if allowInsert:
					# Add the origins to the list of the next level
					for origin in current_cell.origin:
						if origin not in visited:
							next_level.append(origin)

			# Increase the distance and move to the next level
			distance += 1
			current_level = next_level

class HeuristicToCS:
	"""
	Heuristic to Charging Station is a class that stores the distance to a CS in a bifurcation cell.
	It is a first version of the route map. 
	"""
	def __init__(self,cs:ChargingStation,distance:int):
		self.cs=cs
		self.distance=distance

class Cell:    
	"""
	Cell is a class that represents a cell of the grid. It can be a street, a bifurcation or free. 
	It contains a maximun of a car and a CS. 
	When it is a street, it has a velocity and a direct link to the nexts cells. 
	The time (t) is used to ensure that the cars respect the security distance. It is like a snake game. 
	Same t represents the tail of the snake.
	"""

	# factorizable
	ONE=1
	TWO=2
	FREE = 0

	def __init__(self, initial_state):
		self.h2cs=[]
		self.state = initial_state
		self.next_state = None
		# factorizable
		self.neighbors = [[0,0,0],[0,0,0],[0,0,0]]
		self.origin=[]
		self.destination=[]
		self.velocity=0
		self.x=-1
		self.y=-1

		self.car=None
		self.cs=None
		self.t=0
		self.semaphore=[] # if there is a car over then close the cells in list
		self.occupation=0
		self.exponentialOccupation=0
		self.exponentialLastT=0

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
	
	# factorizable
	def updateColor(self):
		count=len(self.origin+self.destination)
		if count==0:
			self.state=Cell.FREE
		elif count==1 or count==2:
			self.state=Cell.ONE
		elif count==2:
			self.state=Cell.TWO

	def update(self,t): 
		# when it is an intersection
		stop=False
		for origin in self.origin: # reversed(self.origin):
			if not stop:
				if origin.car is None:
					continue
				stop=True
			else:
				origin.t=t+1

	def color(self,city):
		cell=self
		
		if cell.state==Cell.FREE:
			return 0
		else:
			if len(cell.destination)==0 or len(cell.origin)==0:
				r=3
			elif len(cell.destination)==2:
				r=2
			else:
				r=1
		
		if cell.t==city.t and cell.car==None:
			r=3
		if cell.car!=None: # and cell.car.id==24:
			r=5
		if cell.cs!=None:
			r=4
		
		# car 0 target is red
		if 0<len(city.cars) and city.cars[0].target==cell:
			r=3

		# artificial origin of semaphores to test location
		# if 0<len(cell.semaphore):
		# 	r=3

		return r

class Street:
	"""
	Street is used as sugar syntax to define a street.

	Attributes:
		path (List[tuple]): List of points of the street
		velocity (int): Velocity of the street
		lanes (int): Number of lanes of the street
	"""
	def __init__(self, path, velocity, lanes, csUbicationable=False):
		self.path = path
		self.velocity = velocity
		self.lanes = lanes
		self.csUbicationable = csUbicationable

class Block:
	"""
	Block is used as sugar syntax to define the streets. 
	The direction of the streets is important because the cars can only move in the direction of the streets.
	At same time you draw the block connet the cells of the grid.
	The construction is a list of streets that is rotated 90 degrees to fill the mosaique of the block.
	"""

	def __init__(self,p):
		self.p=p
		r = 1
		self.lanes=[]
		self.velocities=[]
		self.csUbicable=[]
		self.sugar(
			Street([ (-1,3), (3,3), (3,-1) ], 1,2), # Rotonda
			# parametrizable
			#Street([(r,3), (r,-1)],1,2), # Roundabout
			#Street([(-1,r),(3,r)],1,2),

			Street([(r,47),(r,3)],2,3,False), # Avenues
			Street([(3,r),(47,r)],2,3), 

			Street([(47,15),(r+1-1,15)],1,1,True), # Calles #incorporación
			Street([(r+1-1,36), (47,36) ],1,1,True), 			
			Street([ (15,r+1-1), (15,47) ],1,1,True),
			Street([ (36,47), (36,r+1-1), ],1,1,True),

			# Street([(r+1-1,15),(47,15)],1,1), # inverse the direction
			# Street([(47,36),(r+1-1,36)  ],1,1), 
			# Street([(15,47), (15,r+1-1)  ],1,1),
			# Street([(36,r+1-1), (36,47)  ],1,1),

		)
		self.semaphores=[]
		self.laneWithCS=7
		self.numberOfCS=0
		# self.semaphores=[
		# 	(2,3,1,4),
		# 	(3,3,2,4),
		# ]
		# self.semaphores=[
		# 	(0,3,1,4),
		# 	(1,3,2,4),
		# ]
		# self.semaphores=[ # peor que autosemaforo
		# 	(-1,3,1,4),
		# 	(0,3,2,4),
		# ]

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
		self.ubiqueCSRest=0

	def semaphore(self,grid,x,y):
		''' 
		If a car stays in (x1,y1) then the semaphore is red in (x2,y2)
		'''
		def semaphore2(x1,y1,x2,y2):
			presed=grid.grid[(y1+y)%grid.heigh,(x1+x)%grid.width]
			inred=grid.grid[(y2+y)%grid.heigh,(x2+x)%grid.width]
			presed.semaphore.append(inred)
			if len(presed.semaphore)!=1:
				print("Error: semaphore")

		# for x1,y1,x2,y2 in self.semaphores:
		# 	semaphore2(x1+1,y1+1,x2+1,y2+1)
		# 	semaphore2(y1+1,-x1-1,y2+1,-x2-1)
		# 	semaphore2(-x1-1,-y1-1,-x2-1,-y2-1)
		# 	semaphore2(-y1-1,x1+1,-y2-1,x2+1)

	def pathPluslane(self,path,lane):
		name=["" for i in range(len(path))]
		for i in range(len(path)-1):
			source=path[i]
			target=path[i+1]
			if source[1]==target[1]: #vertical
				if source[0]<target[0]: #up
					key="u"
				else: #down
					key="d"
			else: #horizontal
				if source[1]<target[1]: #right
					key="r"
				else: #left
					key="l"
			name[i]+=key
			name[i+1]+=key
			
		
		switch={				
			"u":(0,1),
			"d":(0,-1),
			"r":(-1,0),
			"l":(1,0),
			"ur":(-1,1),
			"ul":(1,1),
			"dr":(1,1),
			"dl":(-1,1),
			"ru":(-1,1),
			"rd":(-1,-1),
			"lu":(1,1),
			"ld":(1,-1),
		}

		newPath=[]
		for i,p in enumerate(path):
			delta=switch[name[i]]
			newPath.append((p[0]+delta[0]*lane,p[1]+delta[1]*lane))
		return newPath

	def sugar(self,*streets):
		for street in streets:
			for lane in range(street.lanes):
				self.lanes.append(self.pathPluslane(street.path,lane))
				self.velocities.append(street.velocity)
				self.csUbicable.append(street.csUbicationable)

	def draw2(self,grid,lastx,lasty,xx,yy,velocity,csUbicationable):
		if lastx is None:
			return
		if lasty is None:
			return
		last=None

	
		used=False
			
		def csUbication(current):
			nonlocal used
			if 0<=self.ubiqueCSRest and csUbicationable and not used:
				current.cs=ChargingStation(self.p,grid,current)
				grid.cs.append(current.cs)
				self.ubiqueCSRest-=1
				used=True

		if lastx == xx:
			inc = -1
			if lasty < yy:
				inc = 1
			for i in range(lasty,yy+inc,inc):
				current=grid.grid[i%grid.heigh,xx%grid.width]
				# csUbication(current)
				grid.link(last,current,velocity)			
				last=current
				yield
		if lasty == yy:
			inc = -1
			if lastx < xx:
				inc = 1
			for i in range(lastx,xx+inc,inc):
				current=grid.grid[yy%grid.heigh,i%grid.width]
				# csUbication(current)
				grid.link(last,current,velocity)
				last=current
				yield
		used=False
		csUbication(current)

	def draw(self,grid,x,y):
		self.ubiqueCSEach=self.p.numberStationsPerBlock/len(self.lanes)
		for i,lane in enumerate(self.lanes):
			self.ubiqueCSRest+=self.ubiqueCSEach
			csUbicationable=self.csUbicable[i]
			lastx=None
			lasty=None
			for point in lane:
				xx=x+point[0]+1
				yy=y+point[1]+1
				#grid.grid[xx][yy].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i],csUbicationable):
					yield
				lastx=xx
				lasty=yy

			lastx=None
			lasty=None
			for point in lane:
				xx=x+point[1]+1
				yy=y-point[0]-1
				#grid.grid[x+point[1]+1,y-point[0]-1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i],csUbicationable):
					yield
				lastx=xx
				lasty=yy
			lastx=None
			lasty=None
			for point in lane:
				xx=x-point[0]-1
				yy=y-point[1]-1
				#grid.grid[x-point[0]-1,y-point[1]-1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i],csUbicationable):
					yield
				lastx=xx
				lasty=yy
			lastx=None
			lasty=None
			for point in lane:
				xx=x-point[1]-1
				yy=y+point[0]+1
				#grid.grid[x-point[1]-1,y+point[0]+1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i],csUbicationable):
					yield
				lastx=xx
				lasty=yy
		if 0<self.ubiqueCSRest:
			print("There are more CS than expected to ubicate")
		
# separable interfaz y modelo
class City:
	"""
	City is a general holder of the simulation. It encapsules low level details of the graphics representation.
	generators are used to draw the buildings of the city and the simulation. It uses the yield instruction.
	Yield can stop the execution of the container function and can be used recursively.
	"""
	def __init__(self,p):		
		self.p=p
		self.block=Block(p)
		self.grid=Grid(p.verticalBlocks*self.block.height,p.horizontalBlocks*self.block.width)

		self.t=0

		city = self
		self.city_generator = city.generator()
		self.g=city.grid
		self.cars=[]
		#g=Grid(100,100)

		# Animation
		#next(self.city_generator)
		#next(self.city_generator)
		
	def shell(self):
		while True:
			shell=input("Simtravel3> ")
			if shell=="plot":
				self.plot()
			else:
				# split shell
				shell2=shell.split(" ")
				if self.p.__dict__.get(shell2[0])!=None:
					setattr(self.p,shell2[0],int(shell2[1]))

	def plot(self,block=False):
		fig, ax = plt.subplots()

		bounds = [0, 1, 2, 3, 4, 5, 6]
		cmap = colors.ListedColormap(['black',  'green', 'blue','red', 'yellow', 'white'])
		norm = colors.BoundaryNorm(bounds, cmap.N)
	

		def extract_color(cell_obj):
			return cell_obj.color(self)

		img = ax.imshow(np.vectorize(extract_color)(self.g.grid), interpolation='nearest', cmap=cmap, norm=norm)
		self.ani = animation.FuncAnimation(fig, self.update, fargs=(img, self.g.grid, self.g.heigh,self.g.width, ), frames=50,interval=1)

		def on_click(event):
			# Evento al hacer click
			x, y = int(event.xdata), int(event.ydata)  # Convertir las coordenadas a enteros para obtener la posición en la matriz
			print(f"X: {x}, Y: {y}")
			car=self.g.grid[y,x].car
			if car!=None:
				print("id car",car.id)

		fig.canvas.mpl_connect('button_press_event', on_click)

		plt.show(block=block)



	def runWithoutPlot(self, times):
		initial_time = time.time()

		for i in range(times):
			try:
				next(self.city_generator)
				elapsed_time = time.time() - initial_time
				percentage_done = (i + 1) / times * 100

				# Estimación del tiempo total basado en el progreso actual
				total_estimated_time = elapsed_time / (percentage_done / 100)
				estimated_completion_time = initial_time + total_estimated_time

				print(
					f"Progress: {percentage_done:.2f}% End: {time.strftime('%H:%M:%S', time.localtime(estimated_completion_time))} Total: {int(round(total_estimated_time))} seconds  ",
					end="\r"
				)
			except StopIteration:
				print("\ncity_generator has no more items to generate.")
				break
		print()  # Para un salto de línea final después de que el ciclo se complete
		
	
	def update(self,frameNum, img, grid, heigh, width):
		try:
			#for i in range(100):
			next(self.city_generator)
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
				newGrid[i, j] = grid[i, j].color(self)
				# if newGrid[i, j] == Cell.STREET:
				# 	print(i,j)
				
		# initial_states = np.random.choice([Cell.STREET, Cell.FREE], grid.shape[0]*grid.shape[1] , p=[0.2, 0.8])
		# newGrid = np.array([state for state in initial_states]).reshape(grid.shape[0], grid.shape[1])
		img.set_data(newGrid)
		return img,


	def generator(self):
		try:
			# Build city streets
			yieldCada=1000
			if self.p.viewDrawCity:
				yieldCada=1
			yieldI=0
			for i in range(self.p.verticalBlocks):
				for j in range(self.p.horizontalBlocks):
					for _ in self.block.draw(self.grid,i*self.block.height+self.block.height//2,j*self.block.width+self.block.width//2):
						if p.viewDrawCity:
							if yieldCada<=yieldI:
								yieldI=0
								yield
							yieldI+=1
					self.block.semaphore(self.grid,i*self.block.height+self.block.height//2,j*self.block.width+self.block.width//2)

			#numberBlocks=self.p.verticalBlocks*self.p.horizontalBlocks
			numberCars=self.p.numberCars
			numberStations=self.p.numberStations
			numberChargingPerStation=self.p.numberChargingPerStation

     		# Put cs (Charge Stations)
			# self.cs=[]
			# for _ in range(numberStations): #*self.verticalBlocks*self.horizontalBlocks): # number of cs
			# 	self.cs.append(ChargingStation(self.p,self.grid,self.grid.randomStreet(),numberChargingPerStation))
			# 	#if p.viewDrawCity:
			# 	yieldCada=1
			# 	if yieldCada<=yieldI:
			# 		yieldI=0
			# 		yield
			# 	yieldI+=1
			for cs in self.grid.cs:
				cs.insertInRouteMap()


			# Orden and filter cs by p.opmitimizeCSSearch
			for cell in self.grid.grid.flatten():
				if 0<len(cell.h2cs):
					cell.h2cs.sort(key=lambda x: x.distance)
					cell.h2cs=cell.h2cs[:self.p.opmitimizeCSSearch]

			# Put cars
			self.cars=[]
			for id in range(numberCars): # number of cars
				self.cars.append(Car(self.p,id,self.grid,self.grid.randomStreet(),self.grid.randomStreet()))
				if p.viewDrawCity:
					if yieldCada<=yieldI:
						yieldI=0
						yield
					yieldI+=1

			# Simulation
			self.stats=Stats(self.p)
			while True:
				self.t+=1
				self.stats.setT(self.t)
				firstTime=True
				while True:
					if firstTime:
						goPriority=0
					else:
						goPriority=minPriority
					maxPriority=-math.inf
					minPriority=math.inf
					for numcar,car in enumerate(self.cars):
						self.stats.addCarState(numcar,car.state)
						if car.isCharging():
							continue
						if firstTime:
							if car.priority<0:
								car.priority=-car.priority
							car.submove=car.cell.velocity

						if car.priority==goPriority:
							car.moveCar(self.t)
						if car.priority<minPriority and 0<car.priority:
							minPriority=car.priority
						if maxPriority<car.priority:
							maxPriority=car.priority
							
					firstTime=False
					if maxPriority<=0:
						break


				for cs in self.grid.cs:
					cs.moveCS(self.t)

				yieldCada=1
				if yieldCada<=yieldI:
					yieldI=0
					yield
				yieldI+=1

		except Exception as e:  # Esto captura cualquier excepción derivada de la clase base Exception
			print(traceback.format_exc())  # Esto imprime la traza completa del error

class Grid:
	"""
	Grid is a class that represents the grid of the city. It is a matrix of cells.
	It stores the intersections of the city to make a semaphore. 
	Also coinains several utility functions to calculate the distance between two cells, to get a random street, and 
	to link two cells.
	"""
	def __init__(self, heigh, width):
		self.width = width
		self.heigh = heigh
		self.streets=0
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
		self.cs=[]
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
		# If the random is fixed and introduced on cars we can reproduce the same simulation
		while True:
			x = random.randint(0,self.width-1)
			y = random.randint(0,self.heigh-1)
			cell=self.grid[y][x]
			if cell.state!=Cell.FREE and cell.car==None:
				return cell
	def link(self,origin,target,velocity):
		if origin!=None:
			if len(target.origin)==0 and len(target.destination)==0:
				self.streets+=1
			if len(origin.origin)==0 and len(origin.destination)==0:
				self.streets+=1
			target.origin.append(origin)
			origin.destination.append(target)
			target.updateColor()
			origin.updateColor()
			target.velocity=velocity
			origin.velocity=velocity
			if len(target.origin)>1:
				self.intersections.append(target)
			
			#autosemaphore
			if len(target.origin)>1:
				#target.origin[0].semaphore.append(origin)
				#target.origin[0].origin[0].semaphore.append(origin.origin[0])
				for d in target.destination:
					d.semaphore.append(origin)
					
class Car:
	"""
	The car class represents a car of the simulation. The moveCar function is the main function. 
	The car moves from one cell to another. Sometimes it is only one cell, but sometimes 
	there are more than one cell (bifurcation). In this case, the car uses the A* algorithm to find the best path.
	If the car has not enough moves to reach the target, it will try to reach the nearest CS to recharge.
	"""
	def __init__(self,p,id, grid: Grid, cell,target):
		

		self.id=id
		self.state:CarState=CarState.Driving

		self.p=p
		self.grid = grid
		self.cell=cell
		cell.car=self
		self.priority=CarPriority.NoAsigned

		self.target=target
		self.target2=None # if need to recharge

		self.submove=0
		# Change V2 to V3. Why use normal? A normal is a sum of uniform distributions. The normal is not limited to [0,1] but the uniform is. The normal by intervals.
		self.moves=p.carMovesFullDeposity*random.random() 

		# initial moves must be enough to reach the CS at least
		dis,cs=self.localizeCS(cell)	
		if self.moves<dis:
			self.target2=cs.cell
			cell.car=None
			self.cell=cs.cell
			self.enterOnCS()
			#self.moves=dis
		
		self.toCell=[]

		self.targets=[]
		last=self.cell
		for i in range(p.carSizeTarget):
			while True:
				cand=self.grid.randomStreet()
				if cand!=last:
					break
			self.targets.append(cand)
		self.goTargets=0

	def inTarget(self,target):
		return self.cell==target
	
	def localizeCS(self,cell,distance=0):
		if cell.cs!=None:
			if self.p.aStarAddCSTimeSteps:
				# Calculate the ticks (time) to attend the CS
				return (cell.cs.TicksStepsToAttend(),cell.cs)
			else:
				return (0,cell.cs)
		if len(cell.h2cs)==0:
			if len(cell.destination)==1:
				return self.localizeCS(cell.destination[0],distance+1)
			else:
				print("Error: in data structure of CS")
		tupla=None
		for aux in cell.h2cs:
			cand=distance+aux.distance
			if tupla==None or cand<tupla[0]:
				tupla=(cand,aux.cs)
		return tupla
	
		# si es por distancia, solo rellena 1
		# aux=cell.h2cs[0]
		# 
		# return (distance+aux.distance,aux.cs)
	
	def isCharging(self):
		return self.state==CarState.ChargingQueue or self.state==CarState.ChargingSlot

	def checkLegalMove(self,cell,toCell):
		dif=abs(self.cell.x-toCell.x)+abs(self.cell.y-toCell.y)
		if dif!=1:
			print("(",cell.x,",",cell.y,") -> (",toCell.x,",",toCell.y,")")
			print("Error in move, no neighbor")

	def enterOnCS(self):
		if not self.isCharging():
			# enter on CS
			self.cell.car=None
			self.cell=None
			cs=self.target2.cs
			cs.queue.append(self)
			cs.carsInCS+=1
			self.target2.car=None
			self.state=CarState.ChargingQueue

	def calculateRoute(self,cell,t):
		dis,ire=self.aStar(cell,self.target,t)

		if len(ire)==0:
			print("Error: in data structure of A*")
		dis2,_=self.localizeCS(self.target)
		if self.moves<dis+dis2:
			self.state=CarState.ToCharging
			# There are not enough moves, need to recharge in CS first
			dis3,cs=self.localizeCS(cell)
			if self.moves<dis3:
				# There are not enough moves, event with recharge in CS
				# In this version we allow negative moves (energy)
				# We don't have studied the case of cars withou energy and how to recharge them
				pass
			ire=self.aStar(cell,cs.cell,t)[1]
			self.target2=cs.cell
		self.toCell=ire

	def moveCar(self,t):
		if self.inTarget(self.target):
			if self.state==CarState.Destination:
				self.target=self.targets[self.goTargets]
				self.goTargets=(self.goTargets+1)%len(self.targets)
			else:
				self.state=CarState.Destination
				self.cell.t=t
				return 
		if self.inTarget(self.target2):
			# There is no error if pass over target2, because target2 is only set when there is need to recharge	
			self.enterOnCS()
			return
		
		cell=self.cell

		def calculateNext(toCell):
			self.calculateRoute(toCell,t)

		if 1==len(cell.destination):
			toCell=cell.destination[0]
		elif 0<len(self.toCell):
			toCell=self.toCell.pop(0)
		else:
			calculateNext(cell)
			toCell=self.toCell.pop(0)

		if toCell.t==t or toCell.car!=None: 
			cell.occupation+=1
			if 0<self.p.aStarUseCellExponentialWeight:
				cell.exponentialOccupation=cell.exponentialOccupation*math.pow(self.p.aStarUseCellExponentialWeight,t-cell.exponentialLastT)+(1-self.p.aStarUseCellExponentialWeight)
				cell.exponentialLastT=t
			if 1<len(cell.destination):
				self.toCell.insert(0,toCell)
			self.state=CarState.Waiting
			self.priority=-abs( self.priority)
			return
		
		# Execute move
		# identifica si es ilegal, no join
		# self.checkLegalMove(cell,toCell)
		#print("(",cell.x,",",cell.y,") -> (",toCell.x,",",toCell.y,")")
		self.cell = toCell
		cell.car = None
		toCell.car = self
		cell.t=t
		cell.occupation+=1/cell.velocity
		if 0<self.p.aStarUseCellExponentialWeight:
			cell.exponentialOccupation=cell.exponentialOccupation*math.pow(self.p.aStarUseCellExponentialWeight,t-cell.exponentialLastT)+(1-self.p.aStarUseCellExponentialWeight)*1/cell.velocity
			cell.exponentialLastT=t
		for s in cell.semaphore:
			s.t=t

		# Calculate priority
		if len(toCell.destination)==1:
			toCell=toCell.destination[0]
			if self.target2!=None:
				self.state=CarState.ToCharging
			else:
				self.state=CarState.Driving
		else:
			if len(self.toCell)==0:
				calculateNext(toCell)
			toCell=self.toCell[0]
	
		if not self.cell in toCell.origin:
			self.checkLegalMove(self.cell,toCell)
			print("Next error")
		self.priority=toCell.origin.index(self.cell)+1

		# Update energy (moves) and sub-moves (velocity)
		self.moves-=1
		self.submove-=1
		if self.submove==0:
			#print("End")
			# negative priority is used to indicate that the car finished the submove
			self.priority=-abs(self.priority)

	def aStar(self,cell:Cell,target:Cell,t):
		return getattr(self,"aStar"+self.p.aStarMethod)(cell,target,t)

	def aStarDistance(self,cell:Cell,target:Cell,t):
		# Distance version
		# only mark visited if it has more than one destination
		visited=set()
		visited.add(cell)
		opened={}
		for d in cell.destination:
			if len(cell.destination)==1:
				opened[d]=[]
			else:
				opened[d]=[d]
		opened2={}
		distancia=1

		while True:
			# solo se añaden los visited con mas de uno
			for (o,r) in opened.items():
				if len(o.destination)==1:
					opened2[o.destination[0]]=r
				else:
					if o not in visited:
						visited.add(o)
						for d in o.destination:
							r2=r.copy()
							r2.append(d)
							opened2[d]=r2
				if o==target:
					return (distancia,opened[o])
			opened=opened2
			opened2={}
			distancia+=1

	def aStarTime(self,cell:Cell,target:Cell,t):
		# Time version
		visited={}
		opening=[TimeNode(self.p,i) for i in range(self.p.aStarDeep)]
		opening[0].setCell(self.grid,cell,target,0)
		opening[0].decision=[]

		def selectBest():
			'''
			If reached target we can't select
			Only best of best is super-momorized, the others in target are destroyed
			'''
			bestTarget=None
			best=None
			for i in range(0,len(opening)):
				if opening[i].cell==None:
					continue
				if opening[i].cell==target:
					if bestTarget==None:
						bestTarget=opening[i]
					else:
						if bestTarget.heuristic()>opening[i].heuristic():
							bestTarget.cell=None
							bestTarget=opening[i]
					continue
				elif best==None or opening[i].heuristic()<best.heuristic():
					best=opening[i]

			if best==None:
				return bestTarget
			# is valid?
			if bestTarget!=None and bestTarget.heuristic()<best.heuristic():
				return bestTarget
			if best==None:
					print("best is None")
			return best

		def selectWorst():
			worst=None
			for i in range(0,len(opening)):
				if opening[i].cell==None:
					return opening[i]
				if worst==None or opening[i].heuristic()>worst.heuristic():
					worst=opening[i]
			return worst
		
		best=selectBest()
		# movesToStop=0
		#print("Target: (",target.x,",",target.y,")")
		while True:
			# Go through all the targets of the best
			yet=visited.get(best.cell)
			if yet==None or best.time<yet:
				visited[best.cell]=best.time
				for d in best.cell.destination:
					# When the target is reached, it is not necessarily finished				
					worst=selectWorst()
					worst.backup() 

					worst.setCell(self.grid,d,target,best.distance+1)
					if self.p.aStarUseCellAverageVelocity and 0<cell.t:
						if 0<self.p.aStarUseCellExponentialWeight:
							worst.time=best.time+best.cell.exponentialOccupation*math.pow(self.p.aStarUseCellExponentialWeight,t-best.cell.exponentialLastT)
						else:
							worst.time=best.time+cell.occupation/cell.t
					else:
						worst.time=best.time+1/best.cell.velocity
					if 0<self.p.aStarAddRoadCarAsTimeSteps and best.cell.car!=None:
						worst.time+=self.p.aStarAddRoadCarAsTimeSteps

					# clone copy of decision list
					worst.decision=best.decision.copy()
					# if d comes from a bifurcation add

					if len(best.cell.destination)>1 and len(worst.decision)<self.p.aStarCalculateEach:
						worst.decision.append(d)
					
					worst.undoBackupIfWorst()
			# free best
			best.cell=None

			# if reached target we can't select
			# only best of best is super-momorized
			best=selectBest()

			#print("(",best.cell.x,",",best.cell.y,")",best.remainder)

			# stops criterias:
			# if reached target and others are worst
			if best.cell==target:
				stopCriteria=True
				for d in opening:
					if d.heuristic()<best.heuristic():
						stopCriteria=False
						break
				if stopCriteria:
					return best.distance,best.decision
			# by number of moves
			# movesToStop+=1
			# if self.p.aStarStepsPerCar<movesToStop:
			#  	return best.distance,best.decision

class TimeNode:
	'''
	Used by A* algorithm. It is a node of the A* tree.
	'''
	def __init__(self,p,id):
		self.p=p
		self.id=id
		self.cell=None
		self.time=0
		self.decision=None
		self.remainder=0
		self.distance=0
	def backup(self):
		self.backupCell=self.cell
		self.backupTime=self.time
		self.backupDecision=self.decision
		self.backupRemainder=self.remainder
		self.backupHeuristic=self.heuristic()
		self.backupDistance=self.distance
	def undoBackupIfWorst(self):
		if self.backupCell!=None and self.heuristic()>self.backupHeuristic:
			self.cell=self.backupCell
			self.time=self.backupTime
			self.decision=self.backupDecision
			self.remainder=self.backupRemainder
			self.distance=self.backupDistance
	def heuristic(self):
		if self.cell==None:
			return math.inf
		return self.time+self.remainder*self.p.aStarRemainderWeight
	def setCell(self,grid:Grid,cell:Cell,target:Cell,distance):
		self.cell=cell
		self.remainder=grid.distance(cell.x,cell.y,target.x,target.y)
		self.distance=distance

class Stats:
	def __init__(self,p):
		self.p=p
		self.carStateFile=None
		self.carStateFileName=p.statsFileName
		if p.statsFileName=="":
			return
		self.carStateFileName=p.statsFileName+"_carstate.json"

	def setT(self,t):
		if self.carStateFileName=="":
			return
		if self.carStateFile==None:
			# Open file
			self.carStateFile=open(self.carStateFileName,"w")
			self.car=[]
			self.t=0		
		else:
			json.dump(self.car,self.carStateFile)
			self.carStateFile.write("\n")  # New line for each t-step
		self.t=t

	def addCarState(self,num,carState):
		if self.carStateFileName=="":
			return
		while len(self.car)<=num:
			self.car.append({})
		self.car[num]=carState	

	def plot(self):
		# close file
		if self.carStateFile is not None:
			self.carStateFile.close()
			
		# Load data from file
		data_over_time = []
		with open(self.carStateFileName, 'r') as file:
			for line in file:
				data_over_time.append(json.loads(line))

		all_values = list(CarState)

		counts_over_time = {value: [arr.count(value.value) for arr in data_over_time] for value in all_values}

		# Plot using a stacked area plot
		x = range(len(data_over_time))
		prev_values = np.zeros(len(data_over_time))
		for value in all_values:
			current_values = prev_values + np.array(counts_over_time[value])
			plt.fill_between(x, prev_values, current_values, label=f'{value.name}')
			prev_values = current_values

		plt.xlabel('Time Steps')
		plt.ylabel('Number of Vehicles')

		# Put in title verticalBlocks, horizontalBlocks, aStarMethod and aStarAddCSTimeSteps
		plt.title(f'{self.p.verticalBlocks}x{self.p.horizontalBlocks} {self.p.numberCars} cars {self.p.aStarMethod} {"Road" if 0<self.p.aStarAddRoadCarAsTimeSteps else ""} {"CS" if self.p.aStarAddCSTimeSteps else ""}')
		#plt.xticks(ticks=range(len(data_over_time)), labels=[f'{i+1}' for i in range(len(data_over_time))])
		plt.legend(loc='lower left')  

		plt.savefig(self.p.statsFileName+".eps" , format='eps', dpi=300)
		plt.savefig(self.p.statsFileName+".pdf" , format='pdf', dpi=300)
		plt.show()
		print()

if __name__ == '__main__':
	p=Parameters()
	city=City(p)
	#city.shell()

	city.plot(True)
	city.runWithoutPlot(1000)
	city.stats.carStateFile.close()
	#city.stats.plot()

	stats=Stats(p)
	stats.plot()