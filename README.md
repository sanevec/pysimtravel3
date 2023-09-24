Module citybuilder
==================

Classes
-------

`Block()`
:   Block is used as sugar syntax to define the streets. 
    The direction of the streets is important because the cars can only move in the direction of the streets.
    At same time you draw the block connet the cells of the grid.
    The construction is a list of streets that is rotated 90 degrees to fill the mosaique of the block.

    ### Methods

    `draw(self, grid, x, y)`
    :

    `draw2(self, grid, lastx, lasty, xx, yy, velocity)`
    :

    `pathPlusLame(self, path, lame)`
    :

    `sugar(self, *streets)`
    :

`Car(p: citybuilder.Parameters, grid: citybuilder.Grid, cell, target)`
:   The car class represents a car of the simulation. The moveCar function is the main function. 
    The car moves from one cell to another. Sometimes it is only one cell, but sometimes 
    there are more than one cell (bifurcation). In this case, the car uses the A* algorithm to find the best path.
    If the car has not enough moves to reach the target, it will try to reach the nearest CS to recharge.

    ### Methods

    `aStart(self, cell: citybuilder.Cell, target: citybuilder.Cell)`
    :

    `aStartDistance(self, cell: citybuilder.Cell, target: citybuilder.Cell)`
    :

    `aStartTime(self, cell: citybuilder.Cell, target: citybuilder.Cell)`
    :

    `inTarget(self, target)`
    :

    `isCharging(self)`
    :

    `localizeCS(self, cell: citybuilder.Cell, distance=0)`
    :

    `moveCar(self, t)`
    :

`CarPriority(*args, **kwds)`
:   CarPriority is used to define the priority of the execution in the grid.
    Our Cellular Automata is asynchronous. Some cells (with cars) are executed before than others.

    ### Ancestors (in MRO)

    * enum.IntEnum
    * builtins.int
    * enum.ReprEnum
    * enum.Enum

    ### Class variables

    `NoAsigned`
    :

    `NoPriority`
    :

    `Priority`
    :

    `StopedNoPriority`
    :

    `StopedPriority`
    :

`CarState(*args, **kwds)`
:   CarState is used to evaluate the efficiency of the ubication of the charging stations.

    ### Ancestors (in MRO)

    * enum.IntEnum
    * builtins.int
    * enum.ReprEnum
    * enum.Enum

    ### Class variables

    `ChargingQueue`
    :

    `ChargingSlot`
    :

    `Driving`
    :

    `ToCharging`
    :

    `Waiting`
    :

`Cell(initial_state)`
:   Cell is a class that represents a cell of the grid. It can be a street, a bifurcation or free. 
    It contains a maximun of a car and a CS. 
    When it is a street, it has a velocity and a direct link to the nexts cells. 
    The time (t) is used to ensure that the cars respect the security distance. It is like a snake game. 
    Same t represents the tail of the snake.

    ### Class variables

    `FREE`
    :

    `ONE`
    :

    `TWO`
    :

    ### Methods

    `add_neighbor(self, neighbor)`
    :

    `color(self, city)`
    :

    `set_next_state(self)`
    :

    `update(self, t)`
    :

    `updateColor(self)`
    :

    `update_state(self)`
    :

`ChargingStation(p, grid, cell, numberCharging)`
:   The distance and fuel consumption in this version are the same. Astar will have to adapt when this simplification is changed.
    Charging station is a cell that can charge cars. It has a queue of cars and a number of charging slots.
    The chargins statation (CS) alson has a route map to all cells of the city. This route map is used to calculate the distance to the CS.
    
    Attributes:
            p (Parameters): Parameters of the simulation
            grid (Grid): Grid of the city
            cell (Cell): Cell of the grid where the CS is located
            numberCharging (int): Number of charging slots
            queue (List[Car]): Queue of cars
            car (List[Car]): List of cars in the charging slots

    ### Methods

    `insertInRouteMap(self, cell)`
    :   Insert the CS in the route map of the city.
        This a reverse version of the A* algorithm. 
        It is used to calculate the distance to the CSs near on the bifurcation cells.

    `moveCS(self)`
    :

`City(p, block)`
:   City is a general holder of the simulation. It encapsules low level details of the graphics representation.
    generators are used to draw the buildings of the city and the simulation. It uses the yield instruction.
    Yield can stop the execution of the container function and can be used recursively.

    ### Methods

    `generator(self)`
    :

    `plot(self, block=False)`
    :

    `runWithoutPlot(self, times)`
    :

    `shell(self)`
    :

    `update(self, frameNum, img, grid, heigh, width)`
    :

`Grid(heigh, width)`
:   Grid is a class that represents the grid of the city. It is a matrix of cells.
    It stores the intersections of the city to make a semaphore. 
    Also coinains several utility functions to calculate the distance between two cells, to get a random street, and 
    to link two cells.

    ### Methods

    `distance(self, x0, y0, x1, y1)`
    :

    `link(self, origin, target, velocity)`
    :

    `randomStreet(self)`
    :

`HeuristicToCS(cs: citybuilder.ChargingStation, distance: int)`
:   Heuristic to Charging Station is a class that stores the distance to a CS in a bifurcation cell.
    It is a first version of the route map.

`Parameters()`
:   Parameters of the simulation
    
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

`Stats(p)`
:   

    ### Methods

    `addCarState(self, num, carState)`
    :

    `plot(self)`
    :

    `setT(self, t)`
    :

`Street(path, velocity, lames)`
:   Street is used as sugar syntax to define a street.
    
    Attributes:
            path (List[tuple]): List of points of the street
            velocity (int): Velocity of the street
            lames (int): Number of lames of the street

    ### Ancestors (in MRO)

    * builtins.tuple

    ### Instance variables

    `lames`
    :   Alias for field number 2

    `path`
    :   Alias for field number 0

    `velocity`
    :   Alias for field number 1

`TimeNode(p, id)`
:   

    ### Methods

    `backup(self)`
    :

    `heuristic(self)`
    :

    `setCell(self, grid: citybuilder.Grid, cell: citybuilder.Cell, target: citybuilder.Cell, distance)`
    :

    `undoBackupIfWorst(self)`
    :
