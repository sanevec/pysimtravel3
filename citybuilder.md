Module citybuilder
==================

Classes
-------

`Block()`
:   

    ### Methods

    `draw(self, grid, x, y)`
    :

    `draw2(self, grid, lastx, lasty, xx, yy, velocity)`
    :

    `pathPlusLame(self, path, lame)`
    :

    `sugar(self, *streets)`
    :

`Car(p: citybuilder.Parameters, grid: citybuilder.Grid, xy, targetCoordiantes: tuple)`
:   

    ### Methods

    `aStartDistance(self, cell: citybuilder.Cell, target: citybuilder.Cell)`
    :

    `aStartTime(self, cell: citybuilder.Cell, target: citybuilder.Cell)`
    :

    `inTarget(self, target)`
    :

    `localizeCS(self, cell: citybuilder.Cell, distance=0)`
    :

    `moveCar(self, t)`
    :

`Cell(initial_state)`
:   

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

`ChargingStation(p, grid, coordinates, numberCharging)`
:   

    ### Methods

    `insertInRouteMap(self, cell)`
    :

    `moveCS(self)`
    :

`City(p, block)`
:   

    ### Methods

    `generator(self)`
    :

    `plotCity(self)`
    :

    `update(self, frameNum, img, grid, heigh, width)`
    :

`Grid(heigh, width)`
:   

    ### Methods

    `distance(self, x0, y0, x1, y1)`
    :

    `link(self, origin, target, velocity)`
    :

    `randomStreet(self)`
    :

`HeuristicToCS(cs: citybuilder.ChargingStation, distance: int)`
:   

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

`Street(path, velocity, lames)`
:   Street(path, velocity, lames)

    ### Ancestors (in MRO)

    * builtins.tuple

    ### Instance variables

    `lames`
    :   Alias for field number 2

    `path`
    :   Alias for field number 0

    `velocity`
    :   Alias for field number 1

`TimeNode()`
:
