import geopandas as gpd
import pandas as pd

from shapely.geometry import LineString

from openavmkit.utilities.geometry import offset_coordinate_feet, create_geo_rect, create_geo_rect_shape


class SynCity:

  def __init__(self, params):
      self.latitude = params['latitude']
      self.longitude = params['longitude']
      self.crs = params['crs']
      self.units = params["units"]
      self.max_width = params['max_width']
      self.max_height = params['max_height']

      if self.units == "imperial":
        self.base_parcel_size = (150, 100)
      elif self.units == "metric":
        self.base_parcel_size = (45, 30)
      else:
        raise ValueError(f"Invalid units: {self.units}")

      self.width_in_blocks = 0
      self.height_in_blocks = 0

      self.gdf_blocks:gpd.GeoDataFrame|None = None
      self.gdf_roads:gpd.GeoDataFrame|None = None

      self.setup()


  def setup(self):
    # Set basic block size
    self.rows_per_block = 2
    self.parcels_per_row = 5

    self.block_size = (
      self.base_parcel_size[0] * self.rows_per_block,
      self.base_parcel_size[1] * self.parcels_per_row
    )

    self.width_in_blocks = int(self.max_width/self.block_size[0])
    self.height_in_blocks = int(self.max_height/self.block_size[1])
    print(f"Width: {self.width_in_blocks}, Height: {self.height_in_blocks}")

    self.build_grid()


  def build_grid(self):
    self.max_cbd_size_in_blocks = (16, 16)
    self.min_cbd_size_in_blocks = (4, 4)

    self.medium_road_every_n_blocks = 8

    blocks = []
    roads = []

    for x in range(self.width_in_blocks):
      for y in range(self.height_in_blocks):
        block = {
          "x": x,
          "y": y
        }
        blocks.append(block)

    for x in range(self.width_in_blocks):
      x_type = "small"
      if x % self.medium_road_every_n_blocks == 0:
        x_type = "medium"
      road = {
        "x": x,
        "y": -1,
        "road_type": x_type,
        "is_vertical": True,
        "name": _get_street_name(x, -1, True)
      }
      roads.append(road)

    for y in range(self.height_in_blocks):
      y_type = "small"
      if y % self.medium_road_every_n_blocks == 0:
        y_type = "medium"
      road = {
        "x": -1,
        "y": y,
        "road_type": y_type,
        "is_vertical": False,
        "name": _get_street_name(-1, y, False)
      }
      roads.append(road)

    self.gdf_blocks = make_geo_blocks(self, blocks)
    self.gdf_roads = make_geo_roads(self, roads)



# Brazos county facts:
# - CBD block:
#   - 350x375 feet
#   - 5-10 parcels per block
# - Residential parcel:
#   - 150-160 x 50-80 feet
#   - 6-10 parcels per "strip"
#   - 2:3 aspect ratio


def make_geo_roads(city: SynCity, roads: list)->gpd.GeoDataFrame:
  o_x = city.longitude
  o_y = city.latitude

  x_length = city.block_size[0] * city.width_in_blocks
  y_length = city.block_size[1] * city.height_in_blocks

  for road in roads:
    if road["is_vertical"]:
      x = road["x"]
      y = 0
    else:
      x = 0
      y = road["y"]

    x_offset = x * city.block_size[0]
    y_offset = y * city.block_size[1]

    pt_y, pt_x = offset_coordinate_feet(o_y, o_x, y_offset, x_offset)

    lat_1 = pt_y
    lon_1 = pt_x
    lat_2 = pt_y
    lon_2 = pt_x

    if road["is_vertical"]:
      lat_2 += y_length
    else:
      lon_2 += x_length

    geo = LineString([(lon_1, lat_1), (lon_2, lat_2)])
    road["geometry"] = geo

  gdf = gpd.GeoDataFrame(data=roads, geometry="geometry", crs=city.crs)
  return gdf


def make_geo_blocks(city: SynCity, blocks: list)->gpd.GeoDataFrame:
  o_x = city.longitude
  o_y = city.latitude

  x_size = city.block_size[0]
  y_size = city.block_size[1]

  for block in blocks:
    x = block["x"]
    y = block["y"]
    x_offset = x * x_size
    y_offset = y * y_size
    pt_y, pt_x = offset_coordinate_feet(o_y, o_x, y_offset, x_offset)
    geo = create_geo_rect_shape(pt_y, pt_x, x_size, y_size, "nw")
    block["geometry"] = geo

  gdf = gpd.GeoDataFrame(data=blocks, geometry="geometry", crs=city.crs)
  return gdf


def _get_street_name(x: int, y:int, is_vertical: bool):
  if is_vertical:
    return f"{_ordinalize(x+1)} Avenue"
  else:
    return f"{_ordinalize(y+1)} Street"


def _ordinalize(n: int)->str:
  if n % 10 == 1: return "1st"
  if n % 10 == 2: return "2nd"
  if n % 10 == 3: return "3rd"
  return f"{n}th"