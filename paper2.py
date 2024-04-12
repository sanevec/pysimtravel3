# Este fichero deriva del paper1 con la intención de implementar el algoritmo genético
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.colors as mcolors
import random
from collections import namedtuple
import traceback
import math
from enum import IntEnum
import json
import time
import copy
import multiprocessing
import os
import argparse
from typing import List, Tuple, Callable
import itertools
from functools import partial
import datetime


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
		self.verticalBlocks=3
		self.horizontalBlocks=3
		self.numberBlocks=self.verticalBlocks*self.horizontalBlocks
		self.numberStationsPerBlock=1# tipical 1/(numberBlocks), 1, 4

		self.numberStations=self.numberStationsPerBlock*self.numberBlocks
		
		self.percentageEV=0.9

		self.densityCars=0.1
		self.carMovesFullDeposity=27000
		self.stepsToRecharge=960 
		self.carRechargePerTic=self.carMovesFullDeposity/self.stepsToRecharge

		self.introduceCarsInCSToStacionaryState=True

		# A* optimization
		self.opmitimizeCSSearch=3 # bigger is more slow
		self.aStarDeep=100 # Number of positions to search in the aStar algorithm
		self.aStarRemainderWeight=2 #* weight of lineal distance to target to time

		# A* optimization Time
		self.aStarCalculateEach=10 # The aStar calculation is slow, so we can calculate it each n bifurcation cells
		self.aStarTimeOutCacheBifulcation=10 # In the bifulcation cell the calculos of time to CS is valid for n tics

		self.carSizeTarget=20 # The target of each car is a secuence of random cells. This parameter is the size of the secuence

		self.aStarMethod="Time" # Time or Distance
		self.aStarRandomCS=False
		self.aStarCSQueueQuery=0.5 # percentage of EV than use the web to see the queue of the CS (time)
		self.aStarCSReserve=0.5 # percentage of EV than reserve a slot OF THE CSQUEUEQUERY 
		

		# when aStarMethod is Time
		self.aStarAddRoadCarAsTimeSteps=0
		self.aStarUseCellAverageVelocity=True # false=time of the street. works in combination with aStarUseCellExponentialWeight
		self.aStarUseCellExponentialWeight=0.95 #* 0 disable, 0-1 weight of old velocity data
		# self.aStarStepsPerCar=100000 # bigger is more slow, more precision

		# interface parameters
		self.viewWarning = True
		self.viewDrawCity = False
		#self.statsFileName="data/stats_" # paper1
		self.statsFileName="paper2/stats_" 
		self.metastatsFileName="paper2/metastats/"


	def clone(self):
		"""
		Creates a deep copy of the Parameters object.
		"""
		return copy.deepcopy(self)
	
	def metaExperiment(self,**m):
		"""
		Take a map [parameter] -> [values] and generte al cartesian product of the values.
		"""
		# generate an array of index
		index=[0]*len(m)
		keys=list(m.keys())
		r=[] # result
		end=False
		while True:
			# r.append(index.copy())
			p=self.clone()
			p.fileName=""
			p.legendName=""
			for i in range(len(index)):
				setattr(p,keys[i],m[keys[i]][index[i]])
				p.fileName=p.fileName+keys[i]+str(m[keys[i]][index[i]])+"_"
				p.legendName=p.legendName+keys[i]+":"+str(m[keys[i]][index[i]])+" "
			# remove last character
			p.fileName=p.fileName[:-1]
			p.legendName=p.legendName[:-1]
			r.append(p)

			i=0 # index to increment
			while True:
				if i==len(index):
					end=True
					break
				index[i]+=1
				# if reseach the end of the array
				size=len(m[keys[i]])
				if index[i]==size:
					index[i]=0
					i+=1
				else:
					break
			if end:
				break
		return r

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
	Charging = 3
	Queueing = 4
	ToCharging = 5

class CarType(IntEnum):
	'''
	The EV type go to charging station when the battery is low. 
	'''
	EV = 0
	ICEV = 1

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
	def __init__(self,p,grid,cell,numPlugins=-1):
		self.p=p
		self.grid=grid
		self.cell=cell
		if numPlugins == -1:
			# self.numPlugins = p.numberChargingPerStation
			self.numPlugins = p.numberChargersPerBlock
		else:
			self.numPlugins = numPlugins
		cell.cs=self
		# Note: factorizable
		self.queue=[]
		self.carsInCS=0
		self.reserve=[]

		#self.insertInRouteMap(cell)
	def createChargins(self):
		p=self.p
		numberCharging=int(self.numPlugins)
		self.numberCharging=numberCharging
		self.car=[None for i in range(numberCharging)]

	def TicksStepsToAttend(self):
		total=0
		for car in self.queue:
			total+=(car.p.carMovesFullDeposity-car.moves)/car.p.carRechargePerTic
		for car in self.car:
			if car!=None:
				total+=(car.p.carMovesFullDeposity-car.moves)/car.p.carRechargePerTic
		for car in self.reserve:
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
			car.state=CarState.Charging

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
		self.tCache=0 # time in with cached is calculated
		self.timeCache=None # time of the calculation
		self.semaphore=[] # if there is a car over then close the cells in list
		self.occupation=0
		self.exponentialOccupation=0
		self.exponentialLastT=0
		self.contaminationLevel=0

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
			if cell.car.p.type==CarType.EV:
				r=5
			else:
				r=6
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

	def __init__(self,p,allocate_CS = True):
		self.p=p
		self.allocate_CS = allocate_CS
		r = 1
		self.lanes=[]
		self.velocities=[]
		self.csUbicable=[]
		self.sugar(
			Street([ (-1,3), (3,3), (3,-1) ], 1,2), # roundabout
			# parametrizable
			#Street([(r,3), (r,-1)],1,2), # cross
			#Street([(-1,r),(3,r)],1,2),

			Street([(r,47),(r,3)],2,2,False), # Avenues
			Street([(3,r),(47,r)],2,2), 

			Street([(47,15),(r+1-1,15)],1,1,True), # Streets #incorporación
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
			if 0<=self.ubiqueCSRest and csUbicationable and not used and self.allocate_CS:
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
		#if 0<self.ubiqueCSRest:
			#print("There are more CS than expected to ubicate")
		
# separable interfaz y modelo
class City:
	"""
	City is a general holder of the simulation. It encapsules low level details of the graphics representation.
	generators are used to draw the buildings of the city and the simulation. It uses the yield instruction.
	Yield can stop the execution of the container function and can be used recursively.
	"""
	def __init__(self,p,indiv=None):		
		self.p=p
		self.indiv = indiv
		self.block=Block(p,allocate_CS = indiv == None)
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

		#self.valid_coordinates = [(cellX,cellY) for (cellX,cellY) in itertools.product(range(len(self.grid.grid[0])),range(len(self.grid.grid))) if self.grid.grid[cellY][cellX].state != self.grid.grid[cellY][cellX].FREE]    # HACKED
		#print(self.valid_coordinates)
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

		bounds = [0, 1, 2, 3, 4, 5, 6, 7]
		cmap = colors.ListedColormap(['black',  'green', 'blue','red', 'yellow', 'white','orange'])
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



		def runWithoutPlot(self, times, returnFits = False, delta = 0.1, corner_factor = 1, gamma = 0.1, acc = None, dif_matrix = None, wind = None):
			initial_time = time.time()
			next(self.city_generator)
			if returnFits:
				carsAtDestination = 0
				width = len(acc)
				height = len(acc[0])
				P = np.zeros((width+2, height+2, times+1))
				G = np.zeros((width+2, height+2, times+1))
				#keys = [car.id for car in self.cars if car.p.type == CarType.ICEV]
				positions = {car.id: [(car.cell.x,car.cell.y)]*times for car in self.cars if car.p.type == CarType.ICEV and car.cell is not None}
				positionsEV = {car.id: [(car.cell.x,car.cell.y)]*times for car in self.cars if car.p.type == CarType.EV and car.cell is not None}
				positionsEV.update({car.id: [(car.target2.x,car.target2.y)]*times for car in self.cars if car.p.type == CarType.EV and car.cell is None})
				print(len(positions)+len(positionsEV))
				print(len(self.cars))
				velocities = {car.id: [(0,0)]*times for car in self.cars if car.p.type == CarType.ICEV}
				accelerations = {car.id: [(0,0)]*times for car in self.cars if car.p.type == CarType.ICEV}
				def substract_tuples(a,b,factor):
						return ((a[0]-b[0])*factor,(a[1]-b[1])*factor)
				def tuple_norm(a):
						return np.sqrt(a[0]**2+a[1]**2)
				E0 = 0
				f1 = 5.53e-1
				f2 = 1.61e-1
				f3 = -2.89e-3
				f4 = 2.66e-1
				f5 = 5.11e-1
				f6 = 1.83e-1
			for i in range(times):
				try:
						for k in range(width):
							for j in range(height):
								self.grid.grid[k,j].contaminationLevel = P[k,j,i]
						if i>0:
							next(self.city_generator)
							#self.cars[523]
						if returnFits:
							for key in positions:# for c in cars, positions[c]=...
								positions[key][i] = (self.cars[key].cell.x,self.cars[key].cell.y)#[(car.cell.x,car.cell.y) for car in self.cars if car.id == key][0]
								if i>0:
										velocities[key][i] = (substract_tuples(positions[key][i],positions[key][i-1],5/1.8)) # Factor due to Amaro
								if i>1:
										accelerations[key][i] = (substract_tuples(velocities[key][i],velocities[key][i-1],1/1.8))
								vel = tuple_norm(velocities[key][i])
								accel = tuple_norm(accelerations[key][i])
								a,b=positions[key][i]
								G[a,b,i] = max(E0, f1 + f2*vel + f3*vel**2 + f4*accel + f5*accel**2 + f6*vel*accel)
							diffusion_edge = (
								P[0:-2, 1:-1, i] + P[2:, 1:-1, i] +    # Up and down
								P[1:-1, 0:-2, i] + P[1:-1, 2:, i]      # Left and right
							)
							diffusion_corner = (
								P[0:-2, 0:-2, i] + P[2:, 2:, i] +      # Upper left and lower right diagonal
								P[2:, 0:-2, i] + P[0:-2, 2:, i]        # Lower left and upper right diagonal
							)
							P[1:-1, 1:-1, i] = acc * (dif_matrix * P[1:-1, 1:-1, i] + delta / (4 + 4 * corner_factor) * (diffusion_edge + diffusion_corner * corner_factor))
							P[1:-1, 1:-1, i+1] = (1-gamma) * acc * (wind[0][:,:,i]*P[2:, 1:-1, i] + wind[1][:,:,i]*P[:-2, 1:-1, i] + wind[2][:,:,i]*P[1:-1, :-2, i] + wind[3][:,:,i]*P[1:-1, 2:, i] + wind[4][:,:,i]*P[2:, :-2, i] + wind[5][:,:,i]*P[2:, 2:, i] + wind[6][:,:,i]*P[:-2, :-2, i] + wind[7][:,:,i]*P[:-2, 2:, i] + wind[8][:,:,i]*P[1:-1, 1:-1, i] + G[1:-1, 1:-1, i])

							for key in positionsEV:
								if self.cars[key].cell is None:
										positionsEV[key][i] = (self.cars[key].target2.x,self.cars[key].target2.y)
								else:
										positionsEV[key][i] = (self.cars[key].cell.x,self.cars[key].cell.y)
										
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
				if returnFits:
						carsAtDestination += len([car for car in self.cars if car.state == CarState.Destination])
						#print(carsAtDestination, '\n')
			if returnFits:
				P = np.maximum(P, 0)
				max_color_value = np.max(P)#/10
				print(max_color_value)
				if True:#Plot?
						def update_plot(frame_number, zarray, plot, fig, ax):
							for collection in ax.collections:
								collection.remove()
							#ax.collections.clear()
							total_pollution = np.sum(zarray[:, :, frame_number])
							plot.set_data(zarray[:, :, frame_number])
							ax.set_title(f"$t={frame_number}$, $P_{{tot}}={total_pollution:.6f}$")
							x_vals = [a[frame_number][1]-1 for a in positions.values()]
							y_vals = [a[frame_number][0]-1 for a in positions.values()]
							x_vals_EV = [a[frame_number][1]-1 for a in positionsEV.values()]
							y_vals_EV = [a[frame_number][0]-1 for a in positionsEV.values()]
							ax.scatter(x_vals, y_vals, color='green', s=3)
							ax.scatter(x_vals_EV, y_vals_EV, color='blue', s=1)

				# Create figure and axes
				fig, ax = plt.subplots()
				#'''
				#for i in range(len(acc)):
				#    for j in range(len(acc[0])):
				#        if acc[i, j] == 0:
				#            ax.text(j, i, 'X', ha='center', va='center', color='red')
				#'''
				# Choose colorbar scale: 'linear' or 'logarithmic'
				for cs in self.indiv.stations:
					i,j = cs.coordinates
					ax.text(j-1, i-1, 'X', ha='center', va='center', color='blue')
					#ax.text(i, j, 'O', ha='center', va='center', color='blue')
				colorbar_scale = 'logarithmic'  # 'logarithmic' or 'linear'

				# Initial plot
				if colorbar_scale == 'logarithmic':
					norm = mcolors.LogNorm(vmin=0.01, vmax=max_color_value) # Avoid zero in log scale
				else:
					norm = mcolors.Normalize(vmin=0.01, vmax=max_color_value)

				plot = ax.imshow(P[1:-1, 1:-1, 0], cmap="inferno", interpolation="nearest", norm=norm)
				colorbar = fig.colorbar(plot, ax=ax, format='%.2f')

				# Creating the animation
				ani = animation.FuncAnimation(fig, update_plot, times, fargs=(P[1:-1,1:-1,:], plot, fig, ax), interval=200)

				# Save the animation as a GIF
				ani.save("pollution_genetic.gif", writer='imagemagick', fps=5)

				plt.show()
			new_acc = np.ones((width+2, height+2))
			new_acc[1:-1,1:-1] = acc
			return -carsAtDestination/times+P[new_acc==1,:].mean()+P[new_acc==1,:].std(), [0] * len(self.grid.cs)
		
		print() 
		self.stats.close()
		
	
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

	def introduceCarsInCSToStacionaryState(self):
		if not self.p.introduceCarsInCSToStacionaryState:
			return
		# count cars in cs
		carInCS=0
		for car in self.cars:
			if car.isCharging():
				carInCS+=1
		# calculate equilibrium of cars in cs in energy terms
		percentageCarsInCS=1/(self.p.carRechargePerTic/self.p.mediumVelocity+1)
		equilibrium=self.p.numberCars*percentageCarsInCS
		# move cars to equilibrium number
		while carInCS<equilibrium:
			# get random car
			car=self.cars[random.randint(0,len(self.cars)-1)]
			# if car is not in cs
			if not car.isCharging():
				# move car to cs
				cell=car.cell
				_,_,cs=car.localizeCS(cell,self.t)	
				car.target2=cs.cell
				cell.car=None
				car.cell=cs.cell
				car.enterOnCS()
				carInCS+=1

		#if self.p.viewWarning: Hacked 
			#if self.p.numberStations<equilibrium:
			#print("Number of chargers:",self.p.numberChargingPerStation)

	def generator(self):
		#try:
			# Build city streets
		yieldCada=1000
		if self.p.viewDrawCity:
			yieldCada=1
		yieldI=0
		for i in range(self.p.verticalBlocks):
			for j in range(self.p.horizontalBlocks):
				for _ in self.block.draw(self.grid,i*self.block.height+self.block.height//2,j*self.block.width+self.block.width//2):
					if self.p.viewDrawCity:
						if yieldCada<=yieldI:
							yieldI=0
							yield
						yieldI+=1
				self.block.semaphore(self.grid,i*self.block.height+self.block.height//2,j*self.block.width+self.block.width//2)

		# Parameters that depend on the number of streets is calculated here
		p=self.p
		if hasattr(self.p, 'listgenerator'):
			self.listgenerator= [(cellX,cellY) for (cellX,cellY) in itertools.product(range(len(self.grid.grid[0])),range(len(self.grid.grid))) if self.grid.grid[cellY][cellX].state != self.grid.grid[cellY][cellX].FREE]
			self.sizes = (len(self.grid.grid[0]),len(self.grid.grid))
			return

			
		p.numberCars=int(p.densityCars*self.grid.streets)
		# medium velocity of cells in the city
		p.mediumVelocity=self.grid.totalVelocity/self.grid.streets
		p.numberChargers=p.numberChargersPerBlock*p.numberBlocks

		#p.numberChargers=p.numberCars*p.percentageEV*p.mediumVelocity/p.carRechargePerTic*p.energy
		p.energy=p.numberChargers/(p.numberCars*p.percentageEV*p.mediumVelocity/p.carRechargePerTic)
		print("energy",p.energy)
		
		#p.numberChargersPerBlock=p.numberChargers/p.numberBlocks
		p.numberChargingPerStation=p.numberChargers//p.numberStations


		#numberBlocks=self.p.verticalBlocks*self.p.horizontalBlocks
		numberCars=self.p.numberCars
		numberStations=self.p.numberStations
		numberChargingPerStation=self.p.numberChargingPerStation

		self.stats=Stats(self.p)


		if self.indiv != None:       
			for CS in self.indiv.stations:
				if CS.num_chargers == 0:
					print('There should not be any CS with 0 chargers')
					continue
				cell = self.grid.grid[CS.coordinates[1]][CS.coordinates[0]] # WHERE IS NUMBER OF CHARGERS BEING TAKEN INTO ACCOUNT
				currentCS=ChargingStation(p,self.grid,cell,CS.num_chargers)
				self.grid.cs.append(currentCS)

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
			cs.createChargins()
			cs.insertInRouteMap()

		# Orden and filter cs by p.opmitimizeCSSearch
		for cell in self.grid.grid.flatten():
			if 0<len(cell.h2cs):
				cell.h2cs.sort(key=lambda x: x.distance)
				cell.h2cs=cell.h2cs[:self.p.opmitimizeCSSearch]

		# Put cars
		self.cars=[]
		for id in range(numberCars): # number of cars
			p2=self.p.clone()
			if (id+1)/numberCars<self.p.percentageEV:
				p2.type=CarType.EV
			else:
				p2.type=CarType.ICEV

			self.cars.append(Car(p2,id,self.grid,self.grid.randomStreet(),self.grid.randomStreet(),self.t))
			if self.p.viewDrawCity:
				if yieldCada<=yieldI:
					yieldI=0
					yield
				yieldI+=1

		
		
		
		self.introduceCarsInCSToStacionaryState()

		# Simulation
		while True:
			self.t+=1
			self.stats.setT(self.t) # INTERCEPTAR
			firstTime=True
			while True:
				if firstTime:
					goPriority=0
				else:
					goPriority=minPriority
				maxPriority=-math.inf
				minPriority=math.inf
				for numcar,car in enumerate(self.cars):
					self.stats.addCarState(numcar,car.state) # Aquí guardar
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

			for numcs,cs in enumerate(self.grid.cs):
				cs.moveCS(self.t)
				self.stats.addCSQueue(numcs,len(cs.queue))

			yieldCada=1
			if yieldCada<=yieldI:
				yieldI=0
				yield
			yieldI+=1

		#except Exception as e:  # Esto captura cualquier excepción derivada de la clase base Exception
		#    print(traceback.format_exc())  # Esto imprime la traza completa del error

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
		self.totalVelocity=0
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
			if (len(target.origin)==0 and len(target.destination)==0) or (len(origin.origin)==0 and len(origin.destination)==0):
				self.streets+=1
				self.totalVelocity+=velocity
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

class Buscador:
	def __init__(self):
		self.cell=None
		self.father=None
		self.heuristico=0
		self.tiempo=0
		self.open=False
		self.numberChildren=0 # number of open children


class Car:
	"""
	The car class represents a car of the simulation. The moveCar function is the main function. 
	The car moves from one cell to another. Sometimes it is only one cell, but sometimes 
	there are more than one cell (bifurcation). In this case, the car uses the A* algorithm to find the best path.
	If the car has not enough moves to reach the target, it will try to reach the nearest CS to recharge.
	"""
	def __init__(self,p,id, grid: Grid, cell,target,t):
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

		# A percentage of cars have CS Queue Query
		self.csqueuequery=False
		self.csreserve=False
		if self.id<p.aStarCSQueueQuery*p.numberCars:
			self.csqueuequery=True
			if self.id<p.aStarCSQueueQuery*p.aStarCSReserve*p.numberCars:
				self.csreserve=True

		# initial moves must be enough to reach the CS at least
		dis,_,cs=self.localizeCS(cell,t)	

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
	
	def localizeCS(self,cell,t,distance=0): 
		# return distance, timeToAttend, cs
		if self.p.aStarRandomCS:
			# select random CS
			cs=random.choice(self.grid.cs) 
			return (distance,0,cs)
		if cell.cs!=None:
			if self.csqueuequery and self.p.aStarMethod=="Time":
				# Calculate the ticks (time) to attend the CS
				return (distance,cell.cs.TicksStepsToAttend(),cell.cs)
			else:
				return (distance,0,cell.cs)
		if len(cell.h2cs)==0:
			if len(cell.destination)==1:
				return self.localizeCS(cell.destination[0],t,distance+1)
			else:
				print("Error: in data structure of CS")
		tupla=None

		if 0<self.p.aStarTimeOutCacheBifulcation:
			getFromCache=True
			if cell.timeCache==None:
				cell.timeCache=[0]*len(cell.h2cs)
				getFromCache=False
			time=t-cell.tCache
			if self.p.aStarTimeOutCacheBifulcation<time:
				getFromCache=False
		else:
			getFromCache=False
				
		for ind,aux in enumerate(cell.h2cs):
			cand=distance+aux.distance
			heuristic=cand
			if self.p.aStarMethod=="Time":
				if getFromCache:
					heuristic=cell.timeCache[ind]
				else:
					heuristic,_=self.aStar(cell,aux.cs.cell,t)					
					if 0<self.p.aStarTimeOutCacheBifulcation:
						cell.timeCache[ind]=cand
						cell.tCache=t
				
				if self.csqueuequery:
					heuristic+=aux.cs.TicksStepsToAttend()

			#  add tickssteps to attended?
			if tupla==None or heuristic<tupla[1]:
				tupla=(cand,heuristic,aux.cs)
		return tupla
	
		# si es por distancia, solo rellena 1
		# aux=cell.h2cs[0]
		# 
		# return (distance+aux.distance,aux.cs)
	
	def isCharging(self):
		return self.state==CarState.Queueing or self.state==CarState.Charging

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
			self.state=CarState.Queueing
			# remove from reserve
			if self.csreserve:
				try:
					cs.reserve.remove(self)
				except:
					pass
			return True
		return False

	def calculateRoute(self,cell,t):
		dis,ire=self.aStar(cell,self.target,t)

		if self.p.type==CarType.ICEV:
			# If the car is ICEV, it will not need to recharge
			self.toCell=ire
			return 

		if len(ire)==0:
			print("Error: in data structure of A*")
		dis2,_,_=self.localizeCS(self.target,t)
		if self.moves<dis+dis2:
			self.state=CarState.ToCharging
			# There are not enough moves, need to recharge in CS first
			dis3,_,cs=self.localizeCS(cell,t)
			if self.moves<dis3:
				# There are not enough moves, event with recharge in CS
				# In this version we allow negative moves (energy)
				# We don't have studied the case of cars withou energy and how to recharge them
				pass
			ire=self.aStar(cell,cs.cell,t)[1]
			self.target2=cs.cell
			if self.csreserve:
				cs.reserve.append(self)
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
			if self.enterOnCS():
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
				a=math.pow(self.p.aStarUseCellExponentialWeight,t-cell.exponentialLastT)
				b=(1-self.p.aStarUseCellExponentialWeight)
				c=1-a-b
				cell.exponentialOccupation=cell.exponentialOccupation*a+c*1/cell.velocity+b*1
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
			a=math.pow(self.p.aStarUseCellExponentialWeight,t-cell.exponentialLastT)
			# b=(1-self.p.aStarUseCellExponentialWeight)
			# c=1-a-b
			# cell.exponentialOccupation=cell.exponentialOccupation*a+c*1/cell.velocity+b*1/cell.velocity
			d=1-a
			cell.exponentialOccupation=cell.exponentialOccupation*a+d*1/cell.velocity
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
		if not hasattr(self.grid,"aStarPerMillisecond"):
			self.grid.aStarTotal=0
			self.grid.aStarPerMillisecond=0
		since=datetime.datetime.now()
		aux= getattr(self,"aStar"+self.p.aStarMethod)(cell,target,t)
		now=datetime.datetime.now()
		millis=(now-since).microseconds/1000
		self.grid.aStarTotal+=1
		self.grid.aStarPerMillisecond+=millis
		print("millis per aStar:",self.grid.aStarPerMillisecond/self.grid.aStarTotal)
		return aux

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

	def aplicarHijos(self,buscador,father, mejora):
		for b in buscador:
			if b.father==father:
				b.heuristico+=mejora
				b.tiempo+=mejora
				if b.tiempo<0:
					print("Error: negative time")
				self.aplicarHijos(buscador,b,mejora)

	def boorarHijos(self,buscador,padre):
		for b in buscador:
			if b.father==padre:
				b.cell=None
				b.father=None
				b.heuristico=0
				b.tiempo=0
				b.open=False
				b.numberChildren=0
				self.boorarHijos(buscador,b)
				
	def aplicarPadre(self,hijo,padre,mejora):
		if padre==None:
			return
		if padre.mejorHijo==hijo:
			padre.heuristico+=mejora
			self.aplicarPadre(padre,padre.father,mejora)

	def path(self,buscador,lista):
		if buscador.father==None:
			return
		self.path(buscador.father,lista)
		lista.append(buscador.cell)
		

		

	def aStarTimeV2(self,cell:Cell,target:Cell,t,buscadores=100):
		buscador=[Buscador() for _ in range(buscadores)]

		buscador[0].cell=cell

		def tiempoDe(cell):
			currentTime=0
			if self.p.aStarUseCellAverageVelocity and 0<cell.t:
				if 0<self.p.aStarUseCellExponentialWeight:
					a=math.pow(self.p.aStarUseCellExponentialWeight,t-cell.exponentialLastT)
					b=1-a
					currentTime=cell.exponentialOccupation*a+b*1/cell.velocity
				else:
					currentTime=cell.occupation/cell.t
			else:
				currentTime=1/cell.velocity
			return currentTime

		totalTime=0
		totalDistance=0

		while True:
			# Localiza mejor buscador no abierto
			father=None
			for cand in buscador:
				if not cand.open and cand.cell!=None:
					if father==None or father.heuristico>cand.heuristico:
						father=cand

			if father==None:
				try:
					return self.aStarTime(cell,target,t,buscadores*10)
				except RecursionError as e:
					print("Error: profundidad máxima de recursión excedida. V21")

			father.open=True
			current=father.cell

			#buscar próxima celda con bifurcación, sumando heurístico
			currentTimeSegment=father.tiempo+tiempoDe(current)
			while len(current.destination)==1:		
				if current==target:
					# me falta marcar el mejor hijo, para reemplazo
					path=[]
					try:
						self.path(father,path)
					except RecursionError as e:
						print("Error: profundidad máxima de recursión excedida. V22")
					return (currentTimeSegment,path)
					# timeV1,pathV1=self.aStarTimeV1(cell,target,t)
					# # compare path with pathV1 if are different
					# # if len(path)!=len(pathV1):
					# # 	print("Error: path different")
					# # for i in range(len(path)):
					# # 	if path[i]!=pathV1[i]:
					# # 		print("Error: path different")

					# return (timeV1,pathV1)
				current=current.destination[0]
				# comprueba si es target
				currentTimeSegment+=tiempoDe(current)


			for d in current.destination:
				dOrigin=abs(d.x-cell.x)+abs(d.y-cell.y)
				totalDistance+=dOrigin
				totalTime+=currentTimeSegment		
				#print("totalDistance",totalDistance)

				distancia=abs(d.x-target.x)+abs(d.y-target.y)
			
				time2=distancia*totalTime/totalDistance
				heuristico=time2+currentTimeSegment

				esta=None
				for b in buscador:
					if b.cell==d:
						esta=b
						break
				if esta!=None:
					if esta.tiempo>currentTimeSegment:
							# Mueve la cuenta.
							father.numberChildren+=1
							esta.father.numberChildren-=1

							# Encuentra una mejor solución procede a su reemplazo
							#mejora=heuristico-esta.heuristico
							esta.heuristico=heuristico
							esta.tiempo=currentTimeSegment
							esta.father=father
							esta.open=False
							esta.numberChildren=0
							try:
								self.boorarHijos(buscador,esta)
							except RecursionError as e:
								print("Error: profundidad máxima de recursión excedida. V23")
							# if esta.open:
							# 	self.aplicarHijos(buscador,esta,mejora)
				else:
					# si no está inserta, la cabeza y los hijos operas...
					# Localiza candidato a reemplazar
					minimo=None
					for i,b in enumerate(buscador):
						if b.numberChildren==0:
							if minimo==None or b.cell==None or b.heuristico>minimo.heuristico:
								minimo=b
								if b.cell==None:
									break
					if minimo.cell==None or minimo.heuristico>heuristico:
						father.numberChildren+=1
						minimo.cell=d
						if minimo.father!=None:
							minimo.father.numberChildren-=1
						minimo.father=father
						minimo.open=False
						minimo.heuristico=heuristico
						minimo.tiempo=currentTimeSegment
						minimo.numberChildren=0

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
							a=math.pow(self.p.aStarUseCellExponentialWeight,t-best.cell.exponentialLastT)
							b=1-a
							worst.time=best.time+best.cell.exponentialOccupation*a+b*1/best.cell.velocity
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

class TimeNode2:
	'''
	Used by A* algorithm. It is a node of the A* tree.
	This is the version 2, in this version the visited is not unloaded.
	'''
	def __init__(self,p,id):
		self.p=p
		self.id=id
		self.cell=None
		self.time=0
		self.parent=None
		self.childs=0
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
		self.statsFileName=p.statsFileName+p.fileName
		if p.statsFileName=="":
			return
		self.carStateFile=None

	def setT(self,t):
		if self.statsFileName=="":
			return
		if self.carStateFile==None:
			self.t=0		

			self.carStateFile=open(self.statsFileName+"_carstate.json","w")
			self.car=[]
			
			self.csQueueFile=open(self.statsFileName+"_csqueue.json","w")
			self.cs=[]
		else:
			json.dump(self.car,self.carStateFile)
			self.carStateFile.write("\n") 

			json.dump(self.cs,self.csQueueFile)
			self.csQueueFile.write("\n")  
		self.t=t

	def close(self):
		if self.statsFileName=="":
			return
		self.carStateFile.close()
		self.csQueueFile.close()

	def addCarState(self,num,carState):
		if self.statsFileName=="":
			return
		while len(self.car)<=num:
			self.car.append({})
		self.car[num]=carState	

	def addCSQueue(self,num,queue):
		if self.statsFileName=="":
			return
		while len(self.cs)<=num:
			self.cs.append(0)
		self.cs[num]=queue	

	def plotCS(self,view=True):
		# Load data from file
		data_over_time = []
		with open(self.statsFileName+"_csqueue.json", 'r') as file:
			for line in file:
				data_over_time.append(json.loads(line))

		# Calculate standard deviation per line
		std = [np.std(arr) for arr in data_over_time]

		# Plot std
		x = range(len(data_over_time))
		plt.plot(x, std)
		plt.xlabel('Time Steps')
		plt.ylabel('Standard Deviation Queue')

		# Plot using a stacked area plot
		# x = range(len(data_over_time))
		# plt.plot(x, data_over_time)
		# plt.xlabel('Time Steps')
		# plt.ylabel('Number of Vehicles')

		# Put in title verticalBlocks, horizontalBlocks, aStarMethod and aStarCSQueueQuery
		#plt.title(self.p.legendName)
		#plt.xticks(ticks=range(len(data_over_time)), labels=[f'{i+1}' for i in range(len(data_over_time))])
		#plt.legend(loc='lower left')  
		plt.legend(loc='lower right')#, bbox_to_anchor=(1, 1.05))


		#plt.savefig(self.p.statsFileName+self.p.fileName+"_csqueue.eps" , format='eps', dpi=600)
		plt.savefig(self.p.statsFileName+self.p.fileName+"_csqueue.pdf" , format='pdf', dpi=600)
		if view:
			plt.show()
		else:
			plt.close()
		#print()		

	def plot(self,view=True):
		# Load data from file
		data_over_time = []
		with open(self.statsFileName+"_carstate.json", 'r') as file:
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

		# Put in title verticalBlocks, horizontalBlocks, aStarMethod and aStarCSQueueQuery
		#plt.title(self.p.legendName)
		#plt.xticks(ticks=range(len(data_over_time)), labels=[f'{i+1}' for i in range(len(data_over_time))])
		#plt.legend(loc='lower left')  
		plt.legend(loc='lower right')#, bbox_to_anchor=(1, 1.05))

		#plt.savefig(self.p.statsFileName+self.p.fileName +"_carstate.eps" , format='eps', dpi=600)
		plt.savefig(self.p.statsFileName+self.p.fileName+"_carstate.pdf" , format='pdf', dpi=600)
		if view:			
			plt.show()
		else:
			plt.close()

class MetaStats2:
	def __init__(self,labelx,fx,labely,fy,labelz,fz,filter=None,scatter=False):
		# filter return true if the experiment is not included
		self.labelx=labelx
		self.fx=fx
		self.labely=labely
		self.fy=fy
		self.labelz=labelz
		self.fz=fz
		self.filter=filter
		self.scatter=scatter

	def Clone(self):
		return MetaStats2(self.labelx,self.fx,self.labely,self.fy,self.labelz,self.fz,self.filter)

class MetaStats:

	def __init__(self):
		self.ps=cartesianExperiment()

		withOutDistance=lambda p:("Only Time",p.aStarMethod=="Distance")
		withOutTime=lambda p:("Only Distance",p.aStarMethod=="Time")

		mss=[
			MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Standard Deviation Queue",lambda p:self.stdCSQueue(p),
				"Use Cell Exponential Weight",lambda p:str(p.aStarUseCellExponentialWeight),
				withOutDistance),
			MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Standard Deviation Queue",lambda p:self.stdCSQueue(p),
				"Density Cars",lambda p:str(int(p.densityCars*100))+"% "+p.aStarMethod),
			# MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
			# 	"Standard Deviation Queue",lambda p:self.stdCSQueue(p),
			# 	"By strategy",lambda p:p.legendName.replace(" aStarCSQueueQuery:"+str(p.aStarCSQueueQuery),"")),
			MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Standard Deviation Queue",lambda p:self.stdCSQueue(p),
				"Method Study",lambda p:p.aStarMethod),
			MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Standard Deviation Queue",lambda p:self.stdCSQueue(p),
				"Energy Study per number of Chargers per CS STD",lambda p:str(p.numberChargersPerBlock)+(" chargers" if p.numberChargersPerBlock!=1 else " charger"),
				withOutDistance),
			MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Average Queue",lambda p:self.averageCSQueue(p),
				"Energy Study: per number of Chargers per CS AVE",lambda p:str(p.numberChargersPerBlock)+(" chargers" if p.numberChargersPerBlock!=1 else " charger"),
				withOutDistance),
		]
		# Duplicate al MetaStats2 with Proditivity
		mss2=[]
		for ms in mss:
			mss3=ms.Clone()
			mss3.labely="Productivity (Average of Destinantion)"
			mss3.fy=lambda p:self.carState(p,CarState.Destination)
			mss3.labelz=ms.labelz+" PS"
			mss2.append(mss3)
			mss2.append(ms)

		mss2.append(MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Productivity (Average of Destinantion)",lambda p:self.carState(p,CarState.Destination),
				"Method Study",lambda p:p.aStarMethod,
				withOutDistance,True))
				#"Density Cars",lambda p:str(int(p.densityCars*100))+"% "+p.aStarMethod),

		for ms in mss2:
			self.execute(ms)

	def execute(self,ms):
		ps=self.ps
		self.strategy={}
		for id,p in enumerate(ps):
			if ms.filter!=None:
				(nameFilter,filter)=ms.filter(p)
				ms.nameFilter=nameFilter
				if filter:
					continue
			x=ms.fx(p)
			y=ms.fy(p)
			# is is nan
			if y!=y:
				print("nan")
				y=ms.fy(p)
			l=ms.fz(p)
			if l not in self.strategy:
				self.strategy[l]=[]
			self.strategy[l].append((x,y))

		for l in self.strategy:
			x2ys={}
			for (x,y) in self.strategy[l]:
				if x not in x2ys:
					x2ys[x]=[]
				x2ys[x].append(y)
			# calculate average and std
			xs=[]
			ys=[]
			# std=[]
			sortx=sorted(x2ys.keys())
			for x in sortx:
				xs.append(x)
				ys.append(np.average(x2ys[x]))
				# std.append(np.std(x2ys[x]))
			self.strategy[l]=(xs,ys)
		self.plot(ms,False)

	def plot(self,ms,view=True):
		plt.figure(figsize=(12,7))
		plt.rcParams.update({'font.size': 18})
		for s in self.strategy:
			(xs,ys)=self.strategy[s]
			#plt.plot(xs, ys, '-o', label=s)

			plt.scatter(xs, ys, color='red', s=50)  # s es el tamaño de los puntos
			coeficientes = np.polyfit(xs, ys, 1)
			recta = np.poly1d(coeficientes)
			plt.plot(xs, recta(xs), "--", color="gray")
			
			yhat = recta(xs)
			ybar = np.mean(ys)
			ssreg = np.sum((yhat - ybar)**2)
			sstot = np.sum((ys - ybar)**2)
			r2 = ssreg / sstot
			plt.text(0.05, 0.95, f"$R^2 = {r2:.2f}$", transform=plt.gca().transAxes, ha="left", va="top")



		plt.ylabel(ms.labely)
		plt.xlabel(ms.labelx)
		if ms.filter!=None:
			title=ms.labelz+" ("+ms.nameFilter+")"
		else:
			title=ms.labelz

		#plt.title(title)
		plt.legend()
		plt.grid(True)
		plt.tight_layout()

		#plt.savefig("metastats/"+title+".eps" , format='eps', dpi=600)
		# if not exists directory, create it
		dir=self.ps[0].metastatsFileName
		if not os.path.exists(dir):
			os.makedirs(dir)
		plt.savefig(dir+title+".pdf" , format='pdf', dpi=600)
		if view:
			plt.show()
		else:
			plt.close()

	def averageCSQueue(self,p):
		# Load data from file
		data_over_time = []
		with open(p.statsFileName+p.fileName+"_csqueue.json", 'r') as file:
			for line in file:
				data_over_time.append(json.loads(line))
		average = [np.average(arr) for arr in data_over_time]
		average2 = np.average(average)
		return average2
	
	def stdCSQueue(self,p):
		# Load data from file
		data_over_time = []
		with open(p.statsFileName+p.fileName+"_csqueue.json", 'r') as file:
			for line in file:
				data_over_time.append(json.loads(line))
		std = [np.std(arr,ddof=1) for arr in data_over_time]
		average2 = np.average(std)
		return average2
	
	def carState(self,p,state):
		# Load data from file
		data_over_time = []
		fileName=p.statsFileName+p.fileName+"_carstate.json"
		if not os.path.isfile(fileName):
			print("Not exists file: "+fileName)
		with open(fileName, 'r') as file:
			for line in file:
				data_over_time.append(json.loads(line))
		all_values = list(CarState)
		# Number of state over time
		counts_over_time = {value: [arr.count(state) for arr in data_over_time] for value in all_values}
		average = [np.average(arr) for arr in counts_over_time.values()]
		average2 = np.average(average)
		return average2



def cartesianExperiment():
	p=Parameters()
	ps=p.metaExperiment(
		#energy=[0.15,0.3,0.45,0.6,0.75,0.9],
		seed=[12,34,56,78,90],
		numberChargersPerBlock=[1,5,10],
		aStarMethod=["Time","Distance"],
		#aStarCSQueueQuery=[0,0.25,0.5,0.75,1], 
		aStarCSQueueQuery=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], 
		aStarCSReserve=[1,0],
		densityCars=[0.05,0.1],
		aStarUseCellExponentialWeight=[0.95,0.5],
		#reserveCS=[False], # it has been removed because legend is too long
	)
	ps2=[]
	for p in ps:
		ok=True
		if p.aStarMethod=="Distance":
			if p.aStarCSQueueQuery!=0:
				ok=False
			if p.aStarUseCellExponentialWeight!=0.95:
				ok=False
		if ok:
			p.id=len(ps2)
			ps2.append(p)
	return ps2

def experiment(i,view=False,cache=True,indiv=None, returnFits = False, numTimesteps = 100, delta = 0.1, corner_factor = 1, gamma = 0.1, acc = None, dif_matrix = None, wind = None):
	p=cartesianExperiment()[i]

	#if exists file of experiment, skip
	if cache:
		if not view and os.path.isfile(p.statsFileName+p.fileName+"_carstate.json") and os.path.isfile(p.statsFileName+p.fileName+"_csqueue.json"):
			#if size is 0
			if os.path.getsize(p.statsFileName+p.fileName+"_carstate.json")>0 and os.path.getsize(p.statsFileName+p.fileName+"_csqueue.json")>0:
				return

	random.seed(p.seed)

	city=City(p,indiv)

	if view:
		print("Running experiment: "+p.legendName)
		city.plot(True)
	else:
		print("Running experiment: "+p.legendName)
		if returnFits == True:
			global_fit, local_fit = city.runWithoutPlot(numTimesteps, returnFits, delta, corner_factor, gamma, acc, dif_matrix, wind)
			return global_fit, local_fit
		else:
			city.runWithoutPlot(numTimesteps) #HACKed antes valía 1000
		
		stats=Stats(p)
		stats.plotCS(False)
		stats.plot(False)


SimulationResult = namedtuple('SimulationResult', ['global_fit', 'local_fit'])


def run_sim(a):
	sim_cache = a[0]
	indiv = a[1]
	numExperiment = a[2]
	aux=Genetic(numExperiment,simulation_cache=sim_cache).run_simulation(indiv)
	return aux

def initialize_individual(valid_coordinates, num_chargers, distance, lim_distance):
	some_coordinates = [random.choice(valid_coordinates)]
	M = random.randint(1,num_chargers)
	j=0
	while len(some_coordinates)<M:
		aux_coord = random.choice(valid_coordinates)
		if all([distance(aux_coord,y)>=lim_distance and distance(y,aux_coord)>lim_distance for y in some_coordinates]):
			some_coordinates.append(aux_coord)
		j+=1
		if j>10000:
			print('Error: No se pueden obtener las CS.')
			exit()
	limits = sorted(random.choices(range(0, num_chargers + 1), k=M - 1))
	limits = [0] + limits + [num_chargers]
	chargers_per_station = [limits[i] - limits[i - 1] for i in range(1, len(limits))]
	stations = [GChargingStation(some_coordinates[k], chargers_per_station[k]) for k in range(M) if chargers_per_station[k]>0]
	return Individual(stations)


# if __name__ == '__main__':
# 	# list of all experiments
# 	ps = cartesianExperiment()
# 	for (i,p) in enumerate(ps):
# 		print(i,p.legendName)

# 	# view an particular experiment
# 	experiment(406)

# 	# execute in background all experiments
# 	start_time = time.time()  
# 	num_processors = multiprocessing.cpu_count()

# 	ps2=[]
# 	for i in range(0,len(ps),50):
# 		ps2.append(ps[i:i+50])

# 	for i,ps in enumerate(ps2):
# 		with multiprocessing.Pool(num_processors) as pool:
# 			pool.map(experiment, range(len(ps)))
# 		print(f'Finished {i+1}/{len(ps2)}')
		
# 	end_time = time.time()  
# 	duration = end_time - start_time 
# 	print(f'Total time: {duration:.2f} seconds')

# 	# generate the metastats
# 	ms=MetaStats()
		
class GChargingStation:
	def __init__(self, coordinates: Tuple[int, int], num_chargers: int):
		self.coordinates: Tuple[int, int] = coordinates
		self.num_chargers: int = num_chargers

	def __repr__(self) -> str:
		return f"GChargingStation(coordinates={self.coordinates}, num_chargers={self.num_chargers})"
	
class Individual:
	def __init__(self, stations: List[GChargingStation]):
		self.stations = stations  # List of ChargingStation objects

	def __repr__(self) -> str:
		return f"Individual(stations={self.stations})"
	
	def __eq__(self, other):
		if not isinstance(other, Individual):
			return NotImplemented
		return sorted(self.stations, key=lambda x: (x.coordinates, x.num_chargers)) == sorted(other.stations, key=lambda x: (x.coordinates, x.num_chargers))

class Genetic:
	def __init__(self,numExperiment, simulation_cache={}) -> None:
		self.numExperiment = numExperiment
		self.population_size = 4#multiprocessing.cpu_count()
		#self.max_num_stations = 5
		self.num_chargers = 45
		self.num_timesteps = 10
		#self.distance = lambda x,y : np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
		self.lim_distance = 10
		self.num_generations: int = 1
		self.mutation_rate: float = 0.9
		p=cartesianExperiment()[numExperiment]
		p.listgenerator=True
		p.numberChargingPerStation=0
		city1 = City(p)
		g = city1.generator()
		for k in g:
			print(k)
			break
		self.distance = lambda x,y: self.aStarDistance(city1.grid.grid[x[1],x[0]],city1.grid.grid[y[1],y[0]])[0]
		self.valid_coordinates = city1.listgenerator
		(n_rows, n_cols) = city1.sizes
		self.delta = 0.2  # Diffusion parameter
		self.corner_factor = 1#/np.sqrt(2)
		self.gamma = 0.1 # Lost to the atmosphere
		acc = np.zeros((n_rows+2, n_cols+2))
		for x,y in self.valid_coordinates:
			acc[x,y] = 1
		acc[0,:]=1
		acc[-1,:]=1
		acc[:,0]=1
		acc[:,-1]=1
		acc_neig_edge = (
			acc[0:-2, 1:-1] + acc[2:, 1:-1] +
			acc[1:-1, 0:-2] + acc[1:-1, 2:]
		)
		acc_neig_corner = (
			acc[0:-2, 0:-2] + acc[2:, 2:] +
			acc[2:, 0:-2] + acc[0:-2, 2:]
		)
		dif_matrix = (1 - (acc_neig_edge + acc_neig_corner * self.corner_factor) / (4 + 4 * self.corner_factor) * self.delta)
		self.acc = acc[1:-1, 1:-1]
		self.dif_matrix = dif_matrix
		WN = 0.8 * np.ones((n_rows+2, n_cols+2, self.num_timesteps+1))
		WE = 0 * np.ones((n_rows+2, n_cols+2, self.num_timesteps+1))
		sign_WN = np.sign(WN).astype(int)
		sign_WE = np.sign(WE).astype(int)
		displ_N = np.zeros_like(WN)
		displ_S = np.zeros_like(WN)
		displ_E = np.zeros_like(WN)
		displ_W = np.zeros_like(WN)
		displ_NW = np.zeros_like(WN)
		displ_NE = np.zeros_like(WN)
		displ_SW = np.zeros_like(WN)
		displ_SE = np.zeros_like(WN)
		stays = np.ones_like(WN)
		for p in range(1, n_rows+1):
			for q in range(1, n_cols+1):
					displ_N[p,q, :] = acc[p,q] * np.maximum(WN[p, q,:], 0) * (1 - acc[p - 1, q + sign_WE[p,q,:]] * abs(WE[p, q, :])) * acc[p - 1, q]
					displ_S[p,q, :] = acc[p,q] * np.maximum(-WN[p, q,:],0) * (1 - acc[p + 1, q + sign_WE[p,q,:]] * abs(WE[p, q, :])) * acc[p + 1, q]
					displ_E[p,q, :] = acc[p,q] * np.maximum(WE[p, q,:], 0) * (1 - acc[p - sign_WN[p,q,:], q + 1] * abs(WN[p, q, :])) * acc[p, q + 1]
					displ_W[p,q, :] = acc[p,q] * np.maximum(-WE[p, q,:],0) * (1 - acc[p - sign_WN[p,q,:], q - 1] * abs(WN[p, q, :])) * acc[p, q + 1]
					displ_NE[p,q,:] = acc[p,q] * np.maximum(WN[p, q,:], 0) * np.maximum(WE[p, q,:], 0) * acc[p - 1, q + 1]
					displ_NW[p,q,:] = acc[p,q] * np.maximum(WN[p, q,:], 0) * np.maximum(-WE[p, q,:],0) * acc[p - 1, q - 1]
					displ_SE[p,q,:] = acc[p,q] * np.maximum(-WN[p, q,:],0) * np.maximum(WE[p, q,:], 0) * acc[p + 1, q + 1]
					displ_SW[p,q,:] = acc[p,q] * np.maximum(-WN[p, q,:],0) * np.maximum(-WE[p, q,:],0) * acc[p + 1, q - 1]
		stays += -(displ_N + displ_S + displ_E + displ_W + displ_NE + displ_NW + displ_SE + displ_SW)
		self.wind = (displ_N[2:, 1:-1, :], displ_S[:-2, 1:-1, :], displ_E[1:-1, :-2, :], displ_W[1:-1, 2:, :], displ_NE[2:, :-2, :], displ_NW[2:, 2:, :], displ_SE[:-2, :-2, :], displ_SW[:-2, 2:, :], stays[1:-1, 1:-1, :])
		self.simulation_cache = simulation_cache  # Diccionario para almacenar los resultados de las simulaciones
		#if self.max_num_stations > self.num_chargers:
		#    print("The number of CS cannot exceed the number of chargers.")
		#    exit()
	'''
	def CS_allocators(self,set_valid_coordinates):
		some_coordinates = [random.choice(set_valid_coordinates)]
		M = random.randint(1,self.num_chargers)
		for _ in range(1,M):
			aux_coord = [x for x in set_valid_coordinates if all([self.distance(x,y)>=self.lim_distance and self.distance(y,x)>self.lim_distance for y in some_coordinates])]
			if aux_coord==[]:
					print('Error: No se pueden obtener las CS.')
					exit()
			some_coordinates.append(random.choice(aux_coord))  
		limits = sorted(random.choices(range(0, self.num_chargers + 1), k=M - 1))
		limits = [0] + limits + [self.num_chargers]
		chargers_per_station = [limits[i] - limits[i - 1] for i in range(1, len(limits))]
		return [GChargingStation(some_coordinates[k], chargers_per_station[k]) for k in range(M)]
	'''


	def aStarDistance(self,cell:Cell,target:Cell, t=0):
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

	def initialize_population(self) -> List[Individual]:
		population: List[Individual] = []
		for l in range(self.population_size):
			indiv = initialize_individual(self.valid_coordinates,self.num_chargers,self.distance,self.lim_distance)
			population.append(indiv)
			print('Initialized ', l+1)
		return population

	'''
	def initialize_population(self) -> List[Individual]:
		population: List[Individual] = []
		for l in range(self.population_size):
			some_coordinates = [random.choice(self.valid_coordinates)]
			M = random.randint(1,self.num_chargers)
			for _ in range(1,M):
				aux_coord = [x for x in self.valid_coordinates if all([self.distance(x,y)>=self.lim_distance and self.distance(y,x)>self.lim_distance for y in some_coordinates])]
				if aux_coord==[]:
					print('Error: No se pueden obtener las CS.')
					exit()
				some_coordinates.append(random.choice(aux_coord))  
			limits = sorted(random.choices(range(0, self.num_chargers + 1), k=M - 1))
			limits = [0] + limits + [self.num_chargers]
			chargers_per_station = [limits[i] - limits[i - 1] for i in range(1, len(limits))]
			stations = [GChargingStation(some_coordinates[k], chargers_per_station[k]) for k in range(M) if chargers_per_station[k]>0]
			population.append(Individual(stations))
			print('Initialized ', l+1)
		return population
	'''
	def calculate_fitness(self, resul_fit) -> float:
		# Maybe study how to implement the Shannon entropy (future work)
		#simulation_result = self.run_simulation(individual)  # This should return a SimulationResult namedtuple
		result = resul_fit.global_fit + sum(resul_fit.local_fit)/len(resul_fit.local_fit) # Or some other function of these. Note that we want to maximize the productivity and minimize the energy (multiobjective optimization)
		return result

	def run_simulation(self, individual: Individual) -> namedtuple:
		# Unique key for this individual
		key = tuple(sorted((station.coordinates, station.num_chargers) for station in individual.stations))
		# verify whether the result is already in the cache
		if key in self.simulation_cache:
			return (key,self.simulation_cache[key])
		#SimulationResult = namedtuple('SimulationResult', ['global_fit', 'local_fit'])
		# Here you run your simulation
		exp_result = experiment(self.numExperiment,view=False,cache=False, indiv = individual, returnFits = True, numTimesteps = self.num_timesteps, delta = self.delta, corner_factor = self.corner_factor, gamma = self.gamma, acc = self.acc, dif_matrix = self.dif_matrix, wind = self.wind)
		result = SimulationResult(global_fit = exp_result[0], local_fit = exp_result[1]) #TERMINAR DE ARREGLAR
		#indiv = individual.stations
		#M = len(indiv)
		#coordinates = [indiv[k].coordinates for k in range(M)]
		#number_chargers = [indiv[k].num_chargers for k in range(M)]
		#result = SimulationResult(global_fit=abs(np.mean(number_chargers)-5) + np.std(number_chargers), local_fit=[self.distance(A,(41,11))+self.distance(A,(41,83)) for A in coordinates])
		# Save result in cache
		self.simulation_cache[key] = result
		return (key,result)

	def select_parents(self) -> List[Individual]:
		#fitness_scores = [self.calculate_fitness(individual) for individual in self.population]
		selected = sorted([(a,self.calculate_fitness(b)) for (a,b) in zip(self.population, self.fitness_values)], key=lambda x: x[1])[:self.population_size // 2]
		#selected = sorted(zip(self.population, self.fitness_values), key=lambda x: x[1])[:self.population_size // 2]
		return [x[0] for x in selected]

	def crossover(self, parent1, parent2):
		parent_1 = copy.deepcopy(parent1)
		parent_2 = copy.deepcopy(parent2)
		results = zip(parent_1.stations + parent_2.stations, self.run_simulation(parent_1)[1].local_fit + self.run_simulation(parent_2)[1].local_fit)
		results_ordered = [elemento for elemento, _ in sorted(results, key=lambda k: k[1])]
		child_stations = []
		for _ in range(len(results_ordered)):
			auxnum = random.choices(range(len(results_ordered)), weights=[1 / (k + 1) for k in range(len(results_ordered))], k=1)[0]
			aux = results_ordered[auxnum]
			results_ordered.pop(auxnum)
			if all([self.distance(aux.coordinates, CS.coordinates) >= self.lim_distance or self.distance(CS.coordinates, aux.coordinates) >= self.lim_distance for CS in child_stations]) and sum([CS.num_chargers for CS in child_stations]) < self.num_chargers:
					child_stations.append(aux)
		if child_stations == parent_1.stations or child_stations == parent_2.stations:
			if len(child_stations) != 1:
					child_stations.pop()
		return Individual(child_stations)
	
	def mutate(self, individual):
		indiv = copy.deepcopy(individual.stations)
		M = len(indiv)
		if random.random() < self.mutation_rate:
			mutation_index = random.randint(0, M-1)
			new_num_chargers = random.randint(1, self.num_chargers)
			indiv[mutation_index].num_chargers = new_num_chargers
			coords = [indiv[k].coordinates for k in range(M) if k != mutation_index]
			aux_coord = random.choice(self.valid_coordinates)
			j=0
			while any([self.distance(aux_coord,y)<self.lim_distance or self.distance(y,aux_coord)<self.lim_distance for y in coords]):
					if j>10000:
						print('Error: No se pueden obtener las CS.')
						exit()
					aux_coord = random.choice(self.valid_coordinates)
					j+=1
			indiv[mutation_index].coordinates = aux_coord
		return Individual(indiv)

	def renormalization(self, individual):
		indiv = copy.deepcopy(individual.stations)
		M = len(indiv)
		chargers_per_station = [CS.num_chargers for CS in indiv]
		CS_chosen = [0] * M
		for _ in range(self.num_chargers):
			max_index = max(range(M), key=lambda i: chargers_per_station[i] / (CS_chosen[i] + 1))
			CS_chosen[max_index] += 1
		indiv = [GChargingStation(copy.deepcopy(indiv[k].coordinates),CS_chosen[k]) for k in range(M) if CS_chosen[k]!=0]
		return Individual(indiv)

	def find_best_solution(self) -> Individual:
		#fitness_scores = zip(self.population,self.fitness_values)#[(individual, self.calculate_fitness(individual)) for individual in self.population]
		min_individual, min_fitness = min([(a,self.calculate_fitness(b)) for (a,b) in zip(self.population, self.fitness_values)], key=lambda x: x[1])
		return min_individual, min_fitness
	


	def parallel_run_simulations(self, individuals):
		
		# Create a pool of workers
		with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
			# Map the individuals to the worker function
			aux = [(self.simulation_cache,indiv, self.numExperiment) for indiv in individuals]
			results = pool.map(run_sim, aux)
			res_list = []
			for x in results:
				self.simulation_cache[x[0]]=x[1]
				res_list.append(x[1])
		# Return the results
		return res_list

	
	# For example, in calculating fitness for the entire population
	def calculate_population_fitness(self):
		# Use the parallel version instead of individual run_simulation calls
		results = self.parallel_run_simulations(self.population)
		return results

	def run(self) -> None:
		self.population: List[Individual] = self.initialize_population()
		self.fitness_values: List[float] = self.calculate_population_fitness()
		best_solution, best_value = self.find_best_solution()
		print("Best Solution:", best_solution, "\nBest value:", best_value)
		values=[0]*(self.num_generations+1)
		values[0]=best_value
		for generation in range(self.num_generations):
			init_time = time.time()
			print('Generation', generation+1)
			parents = self.select_parents()
			new_population = copy.deepcopy(parents)
			while len(new_population) < self.population_size:
					parent1, parent2 = random.sample(parents, 2)
					newindiv = self.crossover(parent1,parent2)
					newindiv = self.mutate(newindiv)
					newindiv = self.renormalization(newindiv)
					if newindiv not in new_population:
						new_population.append(newindiv)
						print('Generation ', generation+1, ' Individuals ', len(new_population))
			if len(new_population) < self.population_size:
					print(f"Loop terminated at generation: {generation}")
					break
			self.population = new_population
			self.fitness_values = self.calculate_population_fitness()
			best_solution, best_value = self.find_best_solution()
			#print(parents[0]==best_solution)
			#print(best_value, '\n', self.calculate_fitness(parents[0]), '\n\n\n')
			#print(sum([CS.num_chargers for CS in parents[0].stations]))
			print("Best Solution:", best_solution, "\nBest value:", best_value)
			values[generation+1]=best_value
			self.population = new_population
			finish_time = time.time()
			aux = [tuple([(CS.coordinates,CS.num_chargers) for CS in indiv.stations]) for indiv in new_population]
			print(len(aux)-len(set(aux)))
			print('Running time: ', finish_time - init_time, 's')
		best_solution, best_value = self.find_best_solution()
		print("Best Solution:", best_solution, "\nBest value:", best_value)
		print('Total different individials: ', len(self.simulation_cache))
		print('Should be: ', (self.num_generations+1)*self.population_size - self.num_generations * (self.population_size // 2))
		plt.plot(range(self.num_generations+1),values)
		plt.show()



# Assuming the cartesianExperiment, experiment, and MetaStats functions are defined elsewhere

if __name__ == '__main__':
	# Set default values to None
	default_values = {'list': None, 'view': 1, 'run': None, 'all': False, 'stats': False}

	parser = argparse.ArgumentParser(description='Selectively run experiments.')
	parser.add_argument('--list', action='store_true', help='List all experiments')
	parser.add_argument('--view', type=int, help='View a specific experiment by index', default=default_values['view'])
	parser.add_argument('--run', type=int, help='Run a specific experiment by index')
	parser.add_argument('--all', action='store_true', help='Run all experiments in the background')
	parser.add_argument('--stats', action='store_true', help='Generate meta statistics')
	parser.add_argument('--genetic', action='store_true', help='Enable genetic algorithm option')
	#parser.add_argument()

	# 44

	args = parser.parse_args()

	if args.genetic:
		if args.run is None:
			print("If the genetic module is used, use the --run option to specify the experiment.\n")
			exit()
		

	if not any(vars(args).values()):
		parser.print_help()
	else:
		ps = cartesianExperiment()

		if args.list:
			for (i, p) in enumerate(ps):
				print(i, p.legendName)

		if args.view is not None:
			experiment(args.view,True)

		if args.run is not None:
			if not args.genetic:
				experiment(args.run)

		if args.all:
			start_time = time.time()
			num_processors = multiprocessing.cpu_count()

			with multiprocessing.Pool(num_processors) as pool:
				pool.map(experiment, range(len(ps)))

			end_time = time.time()
			duration = end_time - start_time
			print(f'Total time: {duration:.2f} seconds')

		if args.stats:
			ms = MetaStats()
