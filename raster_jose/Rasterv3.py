from osmread import parse_file, Way, Node
import xml.etree.ElementTree as ET

from PersonalLogger import MyLogger

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os
import time
import math

from pyproj import Transformer


class Raster:

    def __init__(self, osmPath, savePath, osmName, sizeSide=3, log=True, showGrid=False):

        self.osmPath: str = osmPath # Path fot the osm file with the city data.
        self.osmName = osmName
        self.osmNameNoExtenseion = osmName.split(".")[0]
        global logger
        logger = MyLogger("test_jose//log//rasterV3", self.osmNameNoExtenseion, log=log)

        assert os.path.exists(self.osmPath), logger(f"No existe el path: {self.osmPath}", "Error")
        self.osmFile = f"{self.osmPath}//{self.osmName}"

        assert os.path.isfile(self.osmFile), logger(f"No esiste el archivo: {self.osmFile}", "Error")

        self.savePath = savePath
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
            # logger(f"Se ha creado el path {self.savePath}", "Info")

        self.sizeSide = sizeSide
        self.showGrid = showGrid
        
        self.inputCRS = "EPSG:4326"
        self.outputCRS = "EPSG:900913"

        self.getBbox() # Get the lon/lat max and min
        self.grid = self.createGrid()
        self.primaryStreets = ['primary', 'trunk', 'motorway', 'motorway_link', 'trunk_link', 'primary_link', 'motorway_junction']
        self.secondaryStreets = ['secondary', 'tertiary', 'residential', 'unclasified', 'secondary_link', 'tertiary_link', 'living_street']

        self.getGraph = False
        self.makeNodes = False

        self.directionNumStr = {
            (1,0): "E",
            (1,1): "SE",
            (1,-1): "NE",
            (0,1): "S",
            (0,-1): "N",
            (-1,0): "O",
            (-1,1): "SO",
            (-1,-1): "NO"
        }

        self.directionStrNum = {
            "E": (1,0),
            "SE": (1,1),
            "NE": (1,-1),
            "S": (0,1),
            "N": (0,-1),
            "O": (-1,0),
            "SO": (-1,1),
            "NO": (-1,-1)
        }

        self.countNodeId = -1
        self.countStreetId = -1
        self.newNodesIdUsed = set()

    def getData(self): 
        """
        Main function to get the data and the tables with the information about the city. Here is where the graph and tables
        are created.
        """       
        globalTime = 0
        msgTime = [f"Tiempo utilizado por sección:"]

        start = time.time()
        self.dfAllNodes = self.getNodes()
        end = time.time()
        globalTime+= end-start
        msgTime.append(f"-Tiempo para obtener los nodos: {end-start}")
        # logger("Se ha realizado la refactorización de la posición de los nodos y se han almacenado en el DF")


        lons = self.dfAllNodes["lon"].values
        lats = self.dfAllNodes["lat"].values
        _xs, _ys = self.coordenateRefactorArray(lons, lats)

        xArray = ((_xs - int(self.rawMinX))/self.sizeSide).astype(np.int32)
        yArray = ((_ys - int(self.rawMinY))/self.sizeSide).astype(np.int32)

        self.dfAllNodes['x'], self.dfAllNodes['y'] = xArray, yArray


        start = time.time()
        self.dfStreets, self.dfStreetsNodes  = self.getStreets() #, self.dfNodesLanes, self.dfLinksLanes

        self.dfStreetsNodes['idStreet'] = self.dfStreetsNodes['idStreet'].astype(str)

        
        if self.makeNodes and not self.getGraph:
            self.dfNewStreetNodes, self.dfNewLaneNodes, self.dfLinksLanes = self.getNodeLanes()
            self.dfStreetsNodes = pd.concat([self.dfStreetsNodes, self.dfNewStreetNodes], ignore_index=True)
            self.dfAllNodes = pd.concat([self.dfAllNodes, self.dfNewLaneNodes], ignore_index=True)


        if self.getGraph:
            matrixNodes = self.saveGraph()

        '''# TODO: Crear aquí una función que itere las calles y añada los nodeLanes al df, tanto nodesAll, dfStreets, dfStreetsNodes.
        # en nodesAll -> Para seguir manteniendo todos los nodos ahí.
        # en dfStreets -> porque son nodos parte de la calle, y como los links van separados, no es "problema" de este df.
        # en dfStreetsNodes -> se añadirá una nueva entrada con la calle y nuevos nodos. Creando NUEVO id para la calle (se tratará como si fuesen distintas)
        # ¿Crear nuevo df que sean links entre calles para usarlos posteriormente y concatenarlos a self.links?'''
        end = time.time()
        globalTime += end - start
        msgTime.append(f"-Tiempo para obtener las calles: {end-start}")


        # logger(f"\n{self.dfAllNodes.to_string}", "Debug")
        # logger(f"\n{self.dfStreets.to_string}", "Debug")
        # logger(f"\n{self.dfStreetsNodes.to_string}", "Debug")

        start = time.time()
        self.dfNewNodes, self.dfLinks = self.getLinks()
        if self.makeNodes and not self.getGraph:
            self.dfLinks = pd.concat([self.dfLinks, self.dfLinksLanes], ignore_index=True)
        end = time.time()
        globalTime += end-start
        msgTime.append(f"- Tiempo para obtener las conexiones entre nodos y los nodos intermedios: {end-start}")


        # TODO: Añadir los carriles, por ahora funciona todo bien y no está dando muchos problemas. Para hacer esto
        # lo mejor sería crear por cada nodo real nodos adyacentes y aplicar el algoritmo de unir puntos.
        # calculo que es más largo: el eje X o el eje Y. Y añadir a -90º el siguiente nodo (a la derecha del movimiento).
        start = time.time()
        self.dfAllNodes = pd.concat([self.dfAllNodes, self.dfNewNodes], ignore_index=True)
        idNodesStreetsUnique = np.concatenate([self.dfStreetsNodes['idNode'].unique(), self.dfNewNodes['id'].values])
        self.dfNodes = self.dfAllNodes[self.dfAllNodes['id'].isin(idNodesStreetsUnique)].reset_index(drop=True)
        # TODO: Obtener los nodos de calle y evitar el resto.
        xArray = self.dfNodes['x']
        yArray = self.dfNodes['y']
        end = time.time()
        globalTime += start-end
        msgTime.append(f"- Tiempo para realizar el filtro de los nodos de calles y obtener sus posiciones: {end-start}")

        # logger(msgTime, "Info")
        # logger(f"Tiempo total utilizado: {globalTime}", "Info")


        # self.saveDataStructureCity()
        pass

        xMyNodes = self.dfNodes.loc[self.dfNodes['id']< 0, ['x']].values
        yMyNodes = self.dfNodes.loc[self.dfNodes['id']< 0, ['y']].values

        if self.showGrid:
            self.grid[xArray, yArray] = 1
            self.grid[xMyNodes, yMyNodes] = 2
            plt.title(self.osmNameNoExtenseion)
            plt.imshow(self.grid, cmap="viridis", interpolation='nearest')
            plt.show()

        print()

    def saveGraph(self):


        self.nodesStreet = self.dfStreetsNodes['idNode'].unique()
        self.dfNodes = self.dfAllNodes.loc[self.dfAllNodes['id'].isin(self.nodesStreet)]
        self.dfNodes.reset_index(drop=True,inplace=True)
        self.dictIndex = {idx: index for index, idx in self.dfNodes['id'].items()}

        lats = self.dfNodes['lat']
        lons = self.dfNodes['lon']

        mercatorX, mercatorY = self.coordenateRefactorArray(lons, lats)
        xArray = ((mercatorX - self.rawMinX)/self.sizeSide)
        yArray = ((mercatorY - self.rawMinY)/self.sizeSide)

        matrixNodes = np.array([xArray, yArray]).T
        matrixEdges = self.getEdges()

        np.save(self.savePath+'//matrix_nodes_no_filter.npy', matrixNodes)
        np.save(self.savePath+'//matrix_edges_no_filter.npy', matrixEdges)

        matrixEdgesO = matrixEdges['origin']
        matrixEdgesD = matrixEdges["destination"]

        components2 = self.findSubGraphLib(matrixEdges)
        comps = self.filterGraph(matrixEdges)
        grid = np.zeros((int(matrixNodes[:,0].max())+1, int(matrixNodes[:, 1].max())+1), dtype=int)

        matrixNodes = matrixNodes[comps]
        newIndex = {oldId: newId for newId, oldId in enumerate(comps)}

        yGrid = matrixNodes[:, 1]
        xGrid = matrixNodes[:, 0]

        orig = matrixEdges['origin'].isin(comps).values
        dest = matrixEdges['destination'].isin(comps).values
        ind = (orig * dest) > 0
        matrixEdges = matrixEdges[ind]
        matrixEdges['origin'] = matrixEdges['origin'].apply(lambda x: newIndex[x]).astype(np.int32)
        matrixEdges['destination'] = matrixEdges['destination'].apply(lambda x: newIndex[x]).astype(np.int32)
        
        # suma = 0
        # for o, d, _, _, _ in matrixEdges.values:
        #     xo, yo = matrixNodes[o]
        #     xd, yd = matrixNodes[d]
        #     suma += (math.sqrt((xd-xo)**2 + (yd-yo)**2))
        # suma = suma/len(matrixEdges)
        # print(suma)
        matrixNodes = matrixNodes.astype(np.int32)
        nextId = len(matrixNodes)
        pointsInterpolates = []
        newEdges = []
        for o, d, a, b, w in matrixEdges.values:
            oPoint = matrixNodes[o]
            dPoint = matrixNodes[d]
            newPoints = self.interpolateNodes(oPoint, dPoint)
            newIds = list(range(nextId, nextId+len(newPoints)-2))
            newIds = [o]+newIds+[d]
            assert len(newPoints)==len(newIds), "Error en tamaños"
            for i in range(len(newPoints)-1):
                p1 = newPoints[i]
                p2 = newPoints[i+1]
                idP1 = newIds[i]
                idP2 = newIds[i+1]

                if idP1 >= nextId:
                    pointsInterpolates.append((p1[0], p1[1]))
                newEdges.append(idP1, idP2, a, b, w)
            nextId = nextId+len(newPoints)-2
            


        grid = grid.T
        grid[yGrid.astype(np.int32), xGrid.astype(np.int32)]=1

        np.save(self.savePath+'//matrix_nodesV2.npy', matrixNodes)
        np.save(self.savePath+'//matrix_edgesV2.npy', matrixEdges)
        return matrixNodes
    
    @staticmethod
    def interpolateNodes(puntoInicial, puntoFinal):
        (x1, y1), (x2, y2) = puntoInicial, puntoFinal

        puntos = []
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy

        while True:
            puntos.append((x1, y1))  # Añadir el punto a la línea
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

        return puntos

    def filterGraph(self, matrixEdges):
        import networkx as nx
        G = nx.DiGraph()

        for o, d, _,_,w in matrixEdges.values:
            G.add_edge(o, d)
            if not w:
                G.add_edge(d, o)
        largest_cc = max(nx.strongly_connected_components(G), key=len)
        H = G.subgraph(largest_cc).copy()

        nodes = H.nodes()
        # Delete not reachable from all
        while True:
            i = 0
            for node in H:
                if i%500==0:
                    print(i)
                descendatns = nx.descendants(H, node)
                if len(descendatns) < len(nodes)-1:
                    print("Tamaño de descendientes: ",len(descendatns), " --- ", i)
                    print("Para el nodo: ", node)
                badNode = node if not nx.descendants(H, node) else None
                if badNode is not None:
                    print("En la posición ", i, " el nodo: ", node, "No tiene descendientes.")
                    nodes.remove(badNode)
                    H = H.subgraph(nodes)
                    break
                i += 1
            if badNode is None:
                break

        reachable_from_all = set.intersection(*(nx.descendants(H, node) | {node} for node in H))
        H = H.subgraph(reachable_from_all).copy()

        nodes = H.nodes()
        # Delete not reachable from all
        while False:
            i = 0
            for node in H:
                if i%500==0:
                    print(i)
                ancestors = nx.ancestors(H, node)
                if len(ancestors) < len(nodes)-1:
                    print("Tamaño de descendientes: ",len(ancestors), " --- ", i)
                    print("Para el nodo: ", node)
                badNode = node if not ancestors else None
                if badNode is not None:
                    print("En la posición ", i, " el nodo: ", node, "No tiene descendientes.")
                    nodes.remove(badNode)
                    H = H.subgraph(nodes)
                    break
                i += 1
            if badNode is None:
                break

        # reachable_to_all = set.intersection(*(nx.ancestors(H, node) | {node} for node in H))
        # H = H.subgraph(reachable_to_all).copy()

        # plt.figure(figsize=(20,20))
        # nx.draw(H, with_labels=True, node_color="skyblue", node_size=500, font_size=10)
        # plt.show()
        
        return list(H.nodes())
    
    def findSubGraphLib(self, matrixEdges):
        import networkx as nx
        G = nx.DiGraph()
        for o, d, _, _, w in matrixEdges.values:
            G.add_edge(o, d)
            if not w:
                G.add_edge(d, o)

        components = list(nx.weakly_connected_components(G))
        for i, comps in enumerate(components):
            print(f"{i}: {len(comps)}")
        return components

    def findSubGraphs(self):

        def dfs(v, visited, adjMatrix):
            visited[v] = True

            for i in range(len(adjMatrix)):
                if (adjMatrix[v,i]==1 or adjMatrix[i,v]==1) and not visited[i]:
                    dfs(i, visited, adjMatrix)

        n = len(self.adjMatrix)
        visited = [False]*n
        components = []
        dictComp = {}

        for v in range(n):
            if not visited[v]:
                dfs(v, visited, self.adjMatrix)
                c = [idx for idx, vis in enumerate(visited) if vis]
                components.append(c)
                dictComp[v] = len(c)
                visited = [False]*n

    # This property generate new Id to nodes, and control de flow of the value.
    @property
    def getNewNodeId(self):
        """Generate and control the news Ids for the nodes

        Returns:
            int: new Node ID, it must be always negative
        """
        ret = self.countNodeId
        self.countNodeId -=1
        # comprueba que el ID no esté en uso
        # while ret not in self.newNodesIdUsed:
        #     ret = self.countNodeId
        #     self.countNodeId -=1
        return ret

    def getInterpolationNodes(self, puntosIniciales, puntosFinales, countIdNode, idNodesInit, idNodesFinish, idStreet):
        """Function to get the interpolates cell between 2 nodes.

        Args:
            puntosIniciales (tuple[int, int]): init point for the interpolation (x, y) position
            puntosFinales (tuple[it, int]): finish point fot the interpolation (x,y) position
            countIdNode (int): the count for the new ids of interpolate nodes.
            idNodesInit (_type_): id of the init node
            idNodesFinish (_type_): id of the finish node
            idStreet (_type_): id of the street of the nodes

        Returns:
            tuple[list[list], list[list], int]: a tuple of 3 diferents variables: list of list for the position of each interpolate node, list of list with the ids for the new nodes, and a int for the final count for new ids.
        """
        # ###CHATGPT CODE### SEGUNDA VERSION
        # v2: Genera puntos para múltiples líneas definidas por listas de puntos iniciales y finales,
        # registrando superposiciones.
        # v3: Añadir funcionalidad de desplazacmiento
        assert len(puntosIniciales) == len(idNodesInit), logger("getInterpolationNodes -> Se han pasado dos tamaños de listas distintas para posiciones e ids init", 'Error')
        assert len(puntosFinales) == len(idNodesFinish), logger("getInterpolationNodes -> Se han pasado dos tamaños de listas distintas para posiciones e ids finish", 'Error')

        lineas = []
        todos_los_puntos = set()  # Conjunto para almacenar todos los puntos generados
        idsLane = []
        msgSuperposition = ["Contador de errores de superposición:"]
        countIdNodeAux = countIdNode
        i = 0
        a = list(zip(puntosIniciales, puntosFinales))
        for pos, ((x1, y1), (x2, y2)) in enumerate(zip(puntosIniciales, puntosFinales)):
            ids = []
            puntos = []
            dx = abs(x2 - x1)
            dy = -abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx + dy

            first = True
            while True:
                if (x1, y1) in todos_los_puntos:
                    if len(self.dfAllNodes[self.dfAllNodes['x']==x1 & self.dfAllNodes['y']==y1].index)>0 and not first:
                        badNodeId = self.dfAllNodes[self.dfAllNodes['x']==x1 & self.dfAllNodes['y']==y1].index
                        # logger(f"Se está generando una superposición de 2 nodos de la calle {idStreet} en {x1, y1}, ids = {badNodeId}")
                    # superposiciones.append((x1, y1))  # Registrar superposición
                    i+=1
                    msgSuperposition.append(f"{i}- En la posición [{x1, y1}]")
                else:
                    todos_los_puntos.add((x1, y1))  # Añadir el punto al conjunto global

                if first:
                    first = False
                    ids.append(idNodesInit[pos])

                else:
                    ids.append(countIdNodeAux)
                    countIdNodeAux -= 1

                puntos.append((x1, y1))  # Añadir el punto a la línea

                if x1 == x2 and y1 == y2:
                    ids[-1] = idNodesFinish[pos]
                    countIdNodeAux += 1
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x1 += sx
                if e2 <= dx:
                    err += dx
                    y1 += sy

            idsLane.append(ids)
            lineas.append(puntos)
        # if len(msgSuperposition)>1: logger(msgSuperposition, "Debug")
        assert len(idsLane[0]) == len(lineas[0]), logger("Al realizar la interpolación no se han obtenido el mismo número de puntos que de ids", 'Error')
        return lineas, idsLane, countIdNodeAux

    def getStreets(self):
        """Function to get the information of the streets from the osm file. 
        The result is a df with the next columns:
        {
            "id": (int)id of the street in the osm file,
            "idNodeInit": (int)id of the node Init,
            "idNodeFinish": (int)id of the node finish,
            "name": (str)name of the street,
            "lanes": (int)number of lanes for this segment of street,
            "lanesB": (int)number of lane for backward side,
            "lanesF": (int)number of lane for forward side,
            "oneway": (bool)flag for the oneway streets,
            "highwayType": (str)type of street (primary, secondary, roundabout)
        }

        Returns:
            _type_: _description_
        """
        streets = []
        streetNodes = []

        listSuperpositionNodes = []

        self.conversorHighway = {
            1: "primary",
            2: "secondary",
            0: "roundabout"
        }

        # logger("Iniciando la obtención de las calles:")
        msgNoSaved = ["Calles no almacenadas por no ser del tiop requerido:"]

        start = time.time()
        for entidad in parse_file(self.osmFile):

            if isinstance(entidad, Way):

                if entidad.tags.get('highway') is None:
                    continue

                if entidad.tags.get("junction")=="roundabout":
                    highwayType = 0
                elif entidad.tags['highway'] in self.primaryStreets:
                    highwayType = 1
                elif entidad.tags['highway'] in self.secondaryStreets:
                    highwayType = 2
                else:
                    msgNoSaved.append(f"La calle con id={entidad.id} tiene un tipo no reconocido de carretera: {entidad.tags['highway']}")
                    continue

                nodesStreet = entidad.nodes
                name = entidad.tags.get("name")
                lanes = int(entidad.tags.get("lanes")) if entidad.tags.get("lanes") is not None else 1
                if highwayType==0:
                    lanes = 2

                lanesF = entidad.tags.get("lanes:forward")
                lanesB = entidad.tags.get("lanes:backward")
                oneway = entidad.tags.get("oneway") == None or entidad.tags.get("oneway") == "yes"

                streetDict = {
                    "id": entidad.id,
                    "idNodeInit": nodesStreet[0],
                    "idNodeFinish": nodesStreet[-1],
                    "name": name if name is not None else np.nan,
                    "lanes": lanes,
                    "lanesB": int(lanesB) if lanesB is not None else np.nan,
                    "lanesF": int(lanesF) if lanesF is not None else np.nan,
                    "oneway": oneway,
                    "highwayType": highwayType,
                }

                for posId, idNode in enumerate(nodesStreet):
                    last = False
                    if idNode not in self.dfAllNodes['id'].values:
                        continue
                    


                    streetNodeDict = {
                        "idStreet": entidad.id,
                        "idNode": idNode,
                        "position": posId
                    }

                    streetNodes.append(streetNodeDict)
                    if posId >=len(nodesStreet)-1:
                        last = True
                        posId -= 1
                    else:
                        if nodesStreet[posId+1] in self.dfAllNodes['id'].values:
                            posId += 1
                        else:
                            posId -=1
                            last = True

                    assert len(self.dfAllNodes.loc[self.dfAllNodes['id']==idNode, ['x','y']].values) < 2, logger("Error, existe más de un nodo con el mismo id", "Error")
                    if len(self.dfAllNodes.loc[self.dfAllNodes['id']==idNode, ['x','y']].values) == 0:
                        continue
                    x1,y1 = self.dfAllNodes.loc[self.dfAllNodes['id']==idNode, ['x','y']].values[0]
                    if nodesStreet[posId] in self.dfAllNodes.loc[self.dfAllNodes['id']==nodesStreet[posId]]:
                        continue
                    if len(self.dfAllNodes.loc[self.dfAllNodes['id']==nodesStreet[posId], ['x','y']].values) == 0:
                        continue
                    x2,y2 = self.dfAllNodes.loc[self.dfAllNodes['id']==nodesStreet[posId], ['x','y']].values[0]

                    assert len(self.dfAllNodes.loc[self.dfAllNodes['id']==idNode, 'highway'].values) < 2, logger(f"Error, existe más de un nodo con el mismo id {idNode, nodesStreet[posId]}", "Error")
                    
                    diff = ((x2-x1)*(-1 * last+ (not last)), (y2-y1)*(-1 * last + (not last)))
                    if diff[0] == 0 and diff[1] == 0:
                        # logger(f"Se están repitiendo 2 nodos de la calle: {entidad.id}: entre los nodos {idNode, nodesStreet[posId]} en {x1, y1}-{x2, y2}")
                        listSuperpositionNodes.append((idNode, nodesStreet[posId], entidad.id))

                    # Esta función sería quizás interesante hacerla separada de este tramo puesto que
                    # tendríamos la información completa a disposición de utilizarla y así trabajar de
                    # forma más precisa.
                    

                streets.append(streetDict)
        end = time.time()

        # logger(msgNoSaved, "Info")
        # logger(f"Fin de la obtención de las calles, tiempo utilizado: {end-start} segundo.", 'Info')
        dfStreets = pd.DataFrame(streets)
        dfNodeStreets = pd.DataFrame(streetNodes)
        # dfNodeLanes = pd.DataFrame(newLaneNodes)
        # dfLinksLanes = pd.DataFrame(linksLanes)

        self.superpositionNodes = listSuperpositionNodes

        return dfStreets, dfNodeStreets#, dfNodeLanes, dfLinksLanes

    def clearNodes(self):
        """
        Delete and correctify the nodes at the same position.

        """
        for n1, n2, idStreet in self.superpositionNodes:
            idStreet = str(idStreet)
            # TODO: Comprobar que para nodos en la misma posición x,y estén en la misma calle y se "unifiquen" en 1 solo nodo, corrigiendo
            # las conexiones. Esto consiste en que si la posición 2,3 están en la misma ubicación, y la 3 no tiene ninguna conexión más, 
            # además del link con la posición 4, se rectifique y se sustituya por la posición 2. usar la función replace puede estar bien.

            nodeS1 = self.dfStreetsNodes[self.dfStreetsNodes['idNode']==n1]
            nodeS2 = self.dfStreetsNodes[self.dfStreetsNodes['idNode']==n2]

            streetsN1 = nodeS1['idStreet']
            streetsN2 = nodeS2['idStreet']

            # Puede ser más de 1, puesto que un nodo puede ser inicio y fin.
            if not nodeS1[nodeS1['idStreet']==idStreet]['position'].values:
                continue
            if not nodeS2[nodeS2['idStreet']==idStreet]['position'].values:
                continue
            positionN1 = nodeS1[nodeS1['idStreet']==idStreet]['position'].values.max()
            positionN2 = nodeS2[nodeS2['idStreet']==idStreet]['position'].values.max()

            if positionN1 > positionN2:
                print(self.dfStreetsNodes[self.dfStreetsNodes['idNode']==n1])
                self.dfStreetsNodes = self.dfStreetsNodes.loc[(self.dfStreetsNodes['idNode']==n1) & (self.dfStreetsNodes['idStreet']!=idStreet),'idNode'].replace(int(n1), int(n2))
                print(self.dfStreetsNodes[self.dfStreetsNodes['idNode']==n1])
                self.dfStreetsNodes = self.dfStreetsNodes[self.dfStreetsNodes['idNode'] != n1]
                print(self.dfStreetsNodes[self.dfStreetsNodes['idNode']==n1])
                # self.dfLinksLanes['origin'].replace(n1,n2)
                # self.dfLinksLanes['destination'].replace(n1, n2)
            else:
                print(self.dfStreetsNodes[self.dfStreetsNodes['idNode']==n2])
                print(self.dfStreetsNodes[(self.dfStreetsNodes['idNode']==n2) & (self.dfStreetsNodes['idStreet']!=idStreet)])
                self.dfStreetsNodes = self.dfStreetsNodes[(self.dfStreetsNodes['idNode'] != n2) & (self.dfStreetsNodes['idStreet']!=idStreet)]
                # self.dfStreetsNodes.loc[(self.dfStreetsNodes['idNode']==n2) & (self.dfStreetsNodes['idStreet']!=idStreet), 'idNode'].replace(int(n2), int(n1))
                print(self.dfStreetsNodes[self.dfStreetsNodes['idNode']==n2])
                self.dfStreetsNodes['idNode'].replace(int(n2), int(n1))
                # self.dfStreetsNodes = self.dfStreetsNodes[self.dfStreetsNodes['idNode'] != n2]
                print(self.dfStreetsNodes[self.dfStreetsNodes['idNode']==n2])
                # self.dfLinksLanes['origin'].replace(n2,n1)
                # self.dfLinksLanes['destination'].replace(n2, n1)

            
            # La idea es ver que ID se reemplaza por qué id. Porque al final solo cambiando eso es más que suficiente, porque aunque esté con otra
            # calle, como aquí no se hacen links (y si se hacen porque tiene más de 1 calle) simplemente el puntero cambiará por el que
            # deseamos.
            
            msg_int = []
            msg_int.append(f"\n\t{self.dfStreetsNodes[self.dfStreetsNodes['idNode']==n1].values} <- {n1}")
            msg_int.append(f"{self.dfStreetsNodes[self.dfStreetsNodes['idNode']==n2].values} <- {n2}")

            # logger(msg_int, "Debug")

    def getNodeLanes(self):
        """Function to get the lane's nodes for the streets with more than 1 lane.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Return 3 differents DataFrames: dfNewStreetNodes for the new nodes created and the pair street,
            dfNewLaneNodes for the new nodes created, dfLinksLanes for the links between the intersections nodes.
        """
        newLaneNodes = []
        linksLanes = []
        streetNodes = []
        listSuperpositionNodes = []
        idStreetLanes = self.dfStreets[self.dfStreets['lanes'] > 1]['id']

        dictDiff2AddNodeStreet = {
            (1,0): (0,1),
            (0,1): (-1,0),
            (-1,0): (0,-1),
            (0,-1): (1, 0)
        }

        dictDiff2AddNodeRoundAbout = {
            (1,0): (0,-1),
            (0,1): (1,0),
            (-1,0): (0,1),
            (0,-1): (-1,0)
        }

        for idStreet in idStreetLanes:
            nodesStreet = self.dfStreetsNodes[self.dfStreetsNodes['idStreet']==str(idStreet)]['idNode'].values
            lanes = self.dfStreets[self.dfStreets['id']==idStreet]['lanes'].values[0]
            highwayType = self.dfStreets[self.dfStreets['id']==idStreet]['highwayType'].values[0]
            for posId,idNode in enumerate(nodesStreet):
                last = False
                if posId >=len(nodesStreet)-1:
                    last = True
                    posId -= 1
                else:
                    if nodesStreet[posId+1] in self.dfAllNodes['id'].values:
                        posId += 1
                    else:
                        posId -=1
                        last = True

                assert len(self.dfAllNodes.loc[self.dfAllNodes['id']==idNode, ['x','y']].values) < 2, logger("Error, existe más de un nodo con el mismo id", "Error")
                if len(self.dfAllNodes.loc[self.dfAllNodes['id']==idNode, ['x','y']].values) == 0:
                    continue
                x1,y1 = self.dfAllNodes.loc[self.dfAllNodes['id']==idNode, ['x','y']].values[0]
                if nodesStreet[posId] in self.dfAllNodes.loc[self.dfAllNodes['id']==nodesStreet[posId]]:
                    continue
                if len(self.dfAllNodes.loc[self.dfAllNodes['id']==nodesStreet[posId], ['x','y']].values) == 0:
                    continue
                x2,y2 = self.dfAllNodes.loc[self.dfAllNodes['id']==nodesStreet[posId], ['x','y']].values[0]

                assert len(self.dfAllNodes.loc[self.dfAllNodes['id']==idNode, 'highway'].values) < 2, logger(f"Error, existe más de un nodo con el mismo id {idNode, nodesStreet[posId]}", "Error")
                highway = self.dfAllNodes.loc[self.dfAllNodes['id']==idNode, 'highway'].values[0]
                
                
                diff = ((x2-x1)*(-1 * last+ (not last)), (y2-y1)*(-1 * last + (not last)))
                if diff[0] == 0 and diff[1] == 0:
                    # logger(f"Se están repitiendo 2 nodos de la calle: {idStreet}: entre los nodos {idNode, nodesStreet[posId]} en {x1, y1}-{x2, y2}")
                    listSuperpositionNodes.append((idNode, nodesStreet[posId], idStreet))
                    continue
                    # TODO: Tener en cuenta si ambos sonn de la misma calle o de distintas.
                # Esta tupla comprueba que eje es el mayor y le devuelve el signo correspondiente. Lo hago
                # de esta forma para simplicidad de código y evitar tantos if.
                diffBool = (
                    (abs(diff[0])>=abs(diff[1]))*(abs(diff[0]/diff[0]) if diff[0] != 0 else 0),
                    (abs(diff[0])<abs(diff[1]))*(abs(diff[1])/diff[1] if diff[1] != 0 else 0)
                )
                if highwayType == 0:
                    addPosition = dictDiff2AddNodeRoundAbout[diffBool]

                else:
                    addPosition = dictDiff2AddNodeStreet[diffBool]

                for l in range(1, lanes):
                    idLaneStreet = f"{idStreet}_{l}"
                    newIdNode = self.getNewNodeId
                    streetNodeDict = {
                        "idStreet": idLaneStreet,
                        "idNode": newIdNode,
                        "position": posId - 1 if posId != -1 else posId
                    }

                    newX, newY = x1+addPosition[0], y1+addPosition[1]
                    newNode = {
                        "id": newIdNode,
                        "lon": np.nan,
                        "lat": np.nan,
                        "highway": highway,
                        "l": l,
                        "x": newX,
                        "y": newY
                    }

                    # This df will be delete, because only the intersection nodes could be linked between themselves.
                    if len(self.dfStreetsNodes[self.dfStreetsNodes['idNode']==idNode].values) > 1:
                        link = {
                            "origin": idNode,
                            "destination": newIdNode,
                            "direction": self.directionNumStr[diffBool],
                            "idStreet": idLaneStreet
                        }

                    streetNodes.append(streetNodeDict)
                    newLaneNodes.append(newNode)
                    linksLanes.append(link)
        dfNewStreetNodes = pd.DataFrame(streetNodes)
        dfNewLaneNodes = pd.DataFrame(newLaneNodes)
        dfLinksLanes = pd.DataFrame(linksLanes)
        return dfNewStreetNodes, dfNewLaneNodes, dfLinksLanes

    def getEdges(self):
        """Function to create edges to a especific data structure. This function was used to get specifics tables for David.

        Returns:
            pd.DataFrame: DataFrame with the information about edges of city graph.
        """
        edges = []
        groupedStreetNodes = self.dfStreetsNodes.groupby("idStreet")['idNode'].agg(list)

        for idStreet, nodes in groupedStreetNodes.items():
            lanes = self.dfStreets.loc[self.dfStreets['id']==idStreet, 'lanes'].values[0]
            oneway = self.dfStreets.loc[self.dfStreets['id']==idStreet, 'oneway'].values[0]
            pass
            for n1, n2 in zip(nodes, nodes[1:]):
                edge = {
                    'origin': self.dictIndex[n1],
                    'destination': self.dictIndex[n2],
                    'vel': 20,
                    'lanes': lanes,
                    'oneway': oneway
                }
                edges.append(edge)
        return pd.DataFrame(edges)

    def getLinks(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Function to create the table of links between nodes, adding interpolated nodes between the raw nodes to obtain the complete roads.
        Is necesary use 2 diferents tables:
        * newNodesDF: table with the news nodes interpolates, which will be concatenated with the raw nodes.
        * dfLinks: Table to link 2 nodes and where is added 2 values with information
        {
            "origin": main node,
            "destination": node where is the destination,
            "direction": direction of the link (N,S,E,O,NE,NO,SE,SO),
            "idStreet": id of the street.
        }


        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 2 DataFrames: first with the newNodes interpolates between raw nodes, and second with the links between each node of streets.
        """

        links = []
        newNodes = []

        groupedStreetNodes = self.dfStreetsNodes.groupby("idStreet")['idNode'].agg(list)
        usedIds = set()
        for idStreet, nodes in groupedStreetNodes.items():
            if isinstance(idStreet, str):
                pass
            nodes = [n for n in nodes if n in self.dfAllNodes['id'].values]
            for idNode1, idNode2 in zip(nodes,nodes[1:]):
                a = self.dfAllNodes.loc[self.dfAllNodes['id']==idNode1]
                b = self.dfAllNodes['id']
                if idNode1 not in self.dfAllNodes['id'].values: continue
                if idNode2 not in self.dfAllNodes['id'].values: continue
                pointInit = self.dfAllNodes.loc[self.dfAllNodes['id']==idNode1, ['x', 'y']].values[0]
                pointFinish = self.dfAllNodes.loc[self.dfAllNodes['id']==idNode2, ['x', 'y']].values[0]

                xyPointsLanes, idsLanes, self.countNodeId = self.getInterpolationNodes([pointInit], [pointFinish], self.countNodeId, [idNode1], [idNode2], idStreet)
                for xyPoints, ids in zip(xyPointsLanes, idsLanes):
                    for pos, ((x1, y1), (x2, y2)) in enumerate(zip(xyPoints, xyPoints[1:])):
                        id1 = ids[pos]
                        id2 = ids[pos+1]

                        if id1 < 0 and id1 not in self.dfAllNodes['id'].values and id1 not in usedIds:
                            node1 = {
                                "id": id1,
                                "lon": np.nan,
                                "lat": np.nan,
                                "highway": "street",
                                "x": x1,
                                "y": y1,
                                "lane": np.nan
                            }
                            newNodes.append(node1)
                            usedIds.add(id1)


                        if id2 not in self.dfAllNodes['id'].values and id2 not in usedIds:
                            node2 = {
                                "id": id2,
                                "lon": np.nan,
                                "lat": np.nan,
                                "highway": "street",
                                "x": x2,
                                "y": y2,
                                "lane": np.nan
                            }

                            newNodes.append(node2)
                            usedIds.add(id2)
                        diff = (x2-x1, y2-y1)

                        link = {
                            "origin": id1,
                            "destination": id2,
                            "direction": self.directionNumStr[diff],
                            "idStreet": idStreet
                        }
                        links.append(link)


        dfNewNodes = pd.DataFrame(newNodes)
        dfLinks = pd.DataFrame(links)
        return dfNewNodes, dfLinks

    def getNodes(self) -> pd.DataFrame:
        """Function to get the raw nodes for Function to retrieve the raw nodes from the OSM file. This is where filters are applied to select only the desired types of nodes and adapt the data to our structure.the osm file. Here is where is used filters to get only the kind of nodes selected, and adapt the data to our structure.
        dfNodes{
            id: node id of osm,
            lon: longitude value, 
            lat: latitud value, 
            highway: type of road, 
            lane: lane position (default: 0)
            }
        Returns:
            dfNodes: pd.DataFrame
        """
        nodes = []

        filterHighway = self.primaryStreets+self.secondaryStreets+["crossing", "traffic_signals"]
        msg = ["Iniciando lectura de nodos"]
        start = time.time()

        node_count = 0
        for entidad in parse_file(self.osmFile):
            if node_count!=0 and node_count%1000==0:
                msg.append(f"- Nodos procesados: {node_count}; Tiempo: {time.time()-start} segundo")

            if isinstance(entidad, Node):
                if not(self.minLon <= entidad.lon <= self.maxLon and self.minLat <= entidad.lat <= self.maxLat):
                    continue
                highway = "street"
                crossing = False
                if "highway" in entidad.tags:
                    highway = entidad.tags["highway"]
                    if highway not in filterHighway:
                        continue

                node = {
                    "id": entidad.id,
                    "lon": entidad.lon,
                    "lat": entidad.lat,
                    "highway": highway,
                    "lane": 0,
                }

                nodes.append(node)
                node_count+=1
        end = time.time()
        dfNodes = pd.DataFrame(nodes)

        # logger([
        #     f"Se ha finalizado la ejecución",
        #     f"-Tiempo: {end-start}.",
        #     f"-Num de nodos: {len(nodes)}.",
        # ], "Info")


        # logger(f"\n{dfNodes.head().to_string()}", "Info")
        return dfNodes

    def saveDataStructureCity(self, originsSave=True):
        """Function to format the data and save it in CSV files. I separated the Destinations and Origins tables because the main table will be Destinations; the Origins table will not be used.

        Args:
            originsSave (bool, optional): Flag to select if save the table with the links to origins. Defaults to True.
        """

        # The table must be a static col size, for this reason if a node has less destination/origin than the max I put -1 instead.
        dictIndex = {idx: index for index, idx in self.dfNodes["id"].items()}
        dictIndex[10000000000] = -1
        maxDestinations = self.dfLinks['destination'].value_counts().values[0]
        maxOrigins = self.dfLinks['origin'].value_counts().values[0]

        # logger(f"Nodo con más destinos: {next(self.dfLinks['destination'].value_counts().items())}", "Info")
        # logger(f"Nodo con más origenes: {next(self.dfLinks['origin'].value_counts().items())}", "Info")

        listDestinations = []
        listOrigins = []

        destinationsGroup = self.dfLinks.groupby("origin")['destination'].agg(list)
        for idNode, dest in destinationsGroup.items():
            dest = dest + [10000000000]*maxDestinations
            dest = [dictIndex[d] for d in dest[:maxDestinations]]
            xDest, yDest = self.dfNodes.loc[self.dfNodes['id']==idNode, ['x','y']].values[0]
            listDestinations.append([xDest, yDest, *dest])

        numpyDestinations = np.array(listDestinations)
        savePath = self.savePath + f"//city_{self.osmName.split('.')[0]}_Destinations.npy"
        np.save(savePath, numpyDestinations)
        # logger(f"Se ha guardado el archivo con los destinos en {savePath}", 'Info')


        if originsSave:
            originsGroup = self.dfLinks.groupby("destination")['origin'].agg(list)
            for idNode, orig in originsGroup.items():
                orig = orig + [10000000000]*maxOrigins
                orig = [dictIndex[o] for o in orig[:maxOrigins]]
                xOrig, yOrig = self.dfNodes.loc[self.dfNodes['id']==idNode, ['x','y']].values[0]
                listOrigins.append([xOrig, yOrig, *orig])

            numpyOrigins = np.array(listOrigins)
            savePath = self.savePath + f"//city_{self.osmName.split('.')[0]}_Origins.npy"
            np.save(savePath, numpyOrigins)
            # logger(f"Se ha guardado el archivo con los orígenes en {savePath}", 'Info')

    def createGrid(self) -> np.array:
        """Generate the city grid

        Returns:
            np.array: 2D matrix that represent the space of the city.
        """
        return np.zeros(shape=(self.xSide+1, self.ySide+1))

    def getBbox(self):
        """Function to get the limit points of the bounding box and transform them to the Mercator projection. Additionally, this function retrieves the sides of the bounding box.
        """

        # Factor de correción de 1/cos(latitud) aproximadamente
        tree = ET.parse(self.osmFile)
        root = tree.getroot()

        bounds = root.find('bounds')
        if bounds is not None:
            self.minLat = float(bounds.get('minlat'))
            self.maxLat = float(bounds.get('maxlat'))
            self.minLon = float(bounds.get('minlon'))
            self.maxLon = float(bounds.get('maxlon'))

            self.latMean = (self.minLat+self.maxLat)/2

            # logger([
            #     "Las latitudes/longitdes máxmas y mínimas son:",
            #     f"\t-minLat: {self.minLat}", f"\t-maxLat: {self.maxLat}", f"\t-minLon: {self.minLon}", f"\t-maxLon: {self.maxLon}"
            #     ], "Info")

            self.rawMinX, self.rawMinY = self.coordenateRefactorPoint(self.minLon, self.minLat)
            self.rawMaxX, self.rawMaxY = self.coordenateRefactorPoint(self.maxLon, self.maxLat)

            self.minX, self.maxX = int(0), int(self.rawMaxX-self.rawMinX) + 1
            self.minY, self.maxY = int(0), int(self.rawMaxY-self.rawMinY) + 1


            logger([
                "Las lon,lat máximas y mínimas pasadas al formato UTM (expresada en metros):",
                f"xLonMin: {self.rawMinX}",
                f"yLatMin: {self.rawMinY}",
                "",
                f"xLonMax: {self.rawMaxX}",
                f"yLatMax: {self.rawMaxY}",
            ], "Info")
            self.xSide = int((self.rawMaxX - self.rawMinX)/self.sizeSide) + 1
            self.ySide = int((self.rawMaxY - self.rawMinY)/self.sizeSide) + 1
            logger([
                "Tamaño del área:",
                f"-Eje X: {self.rawMaxX - self.rawMinX:.2f} metros",
                f"-Eje Y: {self.rawMaxY - self.rawMinY:.2f} metros",
                "",
                f"-Eje X resized: {self.xSide} metros",
                f"-Eje Y resized: {self.ySide} metros",

            ], 'Info')
        else:
            assert False, logger(f"No existe el campo 'bounds' en el archivo osmFile proporcionado {self.osmFile}", "Error")

    def mercatorCorector(self, value, latMean=None) -> float:
        """Function to apply the mercator correction to get the real size.

        Args:
            value (float): value to apply the correction
            latMean (float, optional): Latitud mean to calculate the ratio of correction. Defaults to None.

        Returns:
            float: value with the mercator correction, this variable could be a point or a np.array[float]
        """
        if latMean is None:
            r = 1/math.cos(math.radians(self.latMean))
            return value/r
        r = 1/math.cos(math.radians(latMean))
        return value/r

    def coordenateRefactorArray(self, lons, lats):
        """Function to get the mercator proyection of a point array

        Args:
            lons (array(float)): array with the lons positions of points
            lats (array(float)): array with the lats positions of points

        Returns:
            array(float,float): array with the points in x,y values
        """
        try:

            assert lons.size==lats.size, logger("Los arrays dados de lon y lat no tienen el mismo tamaño")

            transformer = Transformer.from_crs(self.inputCRS, self.outputCRS, always_xy=True)

            msg =[
                f"Calculando la refactorización de los puntos lon/lat del array de tamaño {lons.size}."
            ]
            start = time.time()
            xs, ys = transformer.transform(lons, lats)
            end = time.time()
            msg.append(f"Se ha terminado la refactorización en {end-start:.2f}s")
            # logger(msg, "Info")
            return self.mercatorCorector(xs), self.mercatorCorector(ys)


        except:
            assert False, logger("No se pudo refactorizar las coordenadas", "Error")

    def coordenateRefactorPoint(self, lonPoint: float, latPoint:float) -> tuple[int, int]:
        """Refactor a single point

        Args:
            lonPoint (float): lon value of the point
            latPoint (float): lat value of the point

        Returns:
            tuple[int, int]: tuple with the new x,y position in mercator's proyection
        """
        try:

            msg =[
                f"Calculando la refactorización del punto {lonPoint, latPoint}."
            ]
            transformer = Transformer.from_crs(self.inputCRS, self.outputCRS, always_xy=True)
            logger(msg, "Info")
            xpoint, ypoint = transformer.transform(lonPoint, latPoint)
            return self.mercatorCorector(xpoint), self.mercatorCorector(ypoint)

        except:
            assert False, logger("No se pudo refactorizar las coordenadas", "Error")
