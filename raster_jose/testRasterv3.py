from Rasterv3 import Raster
file_name = "small_cartuja.osm"
file_name = "cartuja_maps.osm"
# file_name = "sevilla.osm"
log = False
showGrid = True
raster = Raster("raster_jose//maps", f"raster_jose//savesRasterV3//{file_name.split('.')[0]}", file_name, log=log, showGrid=showGrid)
raster.getData()
# links = raster.dfLinks
# nodes = raster.dfNodes


# print(nodes.groupby(['x', 'y']).count())
# print(links.groupby('destination')['origin'].count())

