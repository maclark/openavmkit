from typing import Any

import geopandas as gpd
import pandas as pd
import shapely

from shapely.geometry import LineString

from openavmkit.utilities.geometry import offset_coordinate_feet, create_geo_rect, create_geo_rect_shape_km, \
  offset_coordinate_m


class SynCity:

  def __init__(self, params):
      self.latitude = params['latitude']
      self.longitude = params['longitude']
      self.crs = params['crs']
      self.anchor = params['anchor']
      self.units = params["units"]
      self.max_width = params['max_width']
      self.max_height = params['max_height']

      self.cbd_width_in_blocks = params['cbd_width_in_blocks']
      self.cbd_height_in_blocks = params['cbd_height_in_blocks']

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

    self.medium_road_every_n_blocks = 8
    self.large_road_every_n_blocks = 16

    blocks = []
    roads = []

    # Determine bounds of CBD
    cbd_center_x = int(self.width_in_blocks / 2)
    cbd_center_y = int(self.height_in_blocks / 2)
    cbd_start_x = cbd_center_x - int(self.cbd_width_in_blocks / 2)
    cbd_start_y = cbd_center_y - int(self.cbd_height_in_blocks / 2)
    cbd_end_x = cbd_center_x + int(self.cbd_width_in_blocks / 2)
    cbd_end_y = cbd_center_y + int(self.cbd_height_in_blocks / 2)

    # Determine locations of large roads:
    # - around the CBD
    # - at the edges of the city
    # - halfway between the CBD and the edges of the city

    mid_x_w = int(cbd_start_x/2)
    mid_x_e = int((self.width_in_blocks - cbd_end_x)/2)
    mid_x_n = int(cbd_start_x + (self.width_in_blocks - cbd_end_x)/2)
    mid_x_s = int(cbd_start_x + (self.width_in_blocks - cbd_end_x)/2)

    large_road_xs = [0, self.width_in_blocks, cbd_start_x, cbd_end_x, mid_x_w, mid_x_e]
    large_road_ys = [0, self.height_in_blocks, cbd_start_y, cbd_end_y, mid_x_n, mid_x_s]

    # Append blocks
    for x in range(self.width_in_blocks):
      for y in range(self.height_in_blocks):
        in_cbd = False

        medium_roads = ((x % self.medium_road_every_n_blocks == 0) or ((x+1) % self.medium_road_every_n_blocks == 0)) + \
                       ((y % self.medium_road_every_n_blocks == 0) or ((y+1) % self.medium_road_every_n_blocks == 0))

        large_roads = ((x % self.large_road_every_n_blocks == 0) or ((x+1) % self.large_road_every_n_blocks == 0)) + \
                      ((y % self.large_road_every_n_blocks == 0) or ((y+1) % self.large_road_every_n_blocks == 0))

        if medium_roads < 1 and large_roads == 0:
          zoning = "residential"
        elif medium_roads < 2 and large_roads < 2:
          zoning = "mixed_use"
        else:
          zoning = "commercial"
        if (x >= cbd_start_x and x < cbd_end_x) and (y >= cbd_start_y and y < cbd_end_y):
          in_cbd = True
          zoning = "cbd"
        if in_cbd is False and x >= cbd_end_x and y >= cbd_start_y and y < cbd_end_y:
          if zoning in ["residential", "mixed_use", "commercial"]:
            zoning = "industrial"

        # adjacent to CBD
        if (x == cbd_start_x-1 or x == cbd_end_x) and (y == cbd_start_y-1 or y == cbd_end_y):
          if zoning == "residential":
            zoning = "mixed_use"

        block = {
          "x": x,
          "y": y,
          "cbd": in_cbd,
          "zoning": zoning
        }
        blocks.append(block)

    # Append vertical rows
    for x in range(self.width_in_blocks+1):
      x_type = "small"
      if x % self.medium_road_every_n_blocks == 0:
        x_type = "medium"
      if x % self.large_road_every_n_blocks == 0:
        x_type = "large"

      road = {
        "x": x,
        "y": -1,
        "road_type": x_type,
        "is_vertical": True,
        "name": _get_street_name(x, -1, True)
      }
      roads.append(road)

    # Append horizontal rows
    for y in range(self.height_in_blocks+1):
      y_type = "small"
      if y % self.medium_road_every_n_blocks == 0:
        y_type = "medium"
      if y % self.large_road_every_n_blocks == 0:
        y_type = "large"

      road = {
        "x": -1,
        "y": y,
        "road_type": y_type,
        "is_vertical": False,
        "name": _get_street_name(-1, y, False)
      }
      roads.append(road)

    units = "m"
    if self.units == "imperial":
      units = "ft"

    self.gdf_blocks = make_geo_blocks_from_city(self, blocks, units)
    self.gdf_roads = make_geo_roads_from_city(self, roads, units)



# Brazos county facts:
# - CBD block:
#   - 350x375 feet
#   - 5-10 parcels per block
# - Residential parcel:
#   - 150-160 x 50-80 feet
#   - 6-10 parcels per "strip"
#   - 2:3 aspect ratio



def _get_draw_anchor_coords(city: SynCity):
  if city.units == "imperial":
    offset_func = offset_coordinate_feet
  elif city.units == "metric":
    offset_func = offset_coordinate_m

  if city.anchor == "center":
    # coordinate represents the center of the city
    lat, lon = offset_func(city.latitude, city.longitude, -city.max_height/2, -city.max_width/2)
  else:
    raise ValueError(f"Unsupported anchor: {city.anchor}")

  return lat, lon


def make_geo_roads_from_city(city: SynCity, roads: list, units: str)->gpd.GeoDataFrame:

  o_y, o_x = _get_draw_anchor_coords(city)

  x_length = city.block_size[0] * (city.width_in_blocks)
  y_length = city.block_size[1] * (city.height_in_blocks)

  offset_func = None
  if units == "ft":
    offset_func = offset_coordinate_feet
  elif units == "m":
    offset_func = offset_coordinate_m

  for road in roads:
    if road["is_vertical"]:
      x = road["x"]
      y = 0
    else:
      x = 0
      y = road["y"]

    print(f" road: {road['name']}, x: {x}, y: {y}")

    x_offset = x * city.block_size[0]
    y_offset = (y-1) * city.block_size[1]

    pt_y, pt_x = offset_func(o_y, o_x, y_offset, x_offset)

    lat_1 = pt_y
    lon_1 = pt_x

    if road["is_vertical"]:
      lat_2, lon_2 = offset_func(pt_y, pt_x, y_length, 0)
    else:
      lat_2, lon_2 = offset_func(pt_y, pt_x, 0, x_length)

    geo = LineString([(lon_1, lat_1), (lon_2, lat_2)])
    road["geometry"] = geo

  gdf = gpd.GeoDataFrame(data=roads, geometry="geometry", crs=city.crs)
  return gdf


def make_geo_blocks_from_city(city: SynCity, blocks: list, units: str)->gpd.GeoDataFrame:
  o_y, o_x = _get_draw_anchor_coords(city)
  return make_geo_blocks(o_y, o_x, city.block_size[1], city.block_size[0], blocks, units, city.crs)


def make_geo_blocks(latitude, longitude, block_size_y, block_size_x, blocks: list, units: str, crs: Any)->gpd.GeoDataFrame:
  blocks = make_geo_blocks_raw(latitude, longitude, block_size_y, block_size_x, blocks, units)
  gdf = gpd.GeoDataFrame(data=blocks, geometry="geometry", crs=crs)
  return gdf


def make_geo_blocks_raw(latitude, longitude, block_size_y, block_size_x, blocks: list, units: str)->list:
  o_y = latitude
  o_x = longitude

  y_size = block_size_y
  x_size = block_size_x

  if units == "ft":
    km_per_ft = 0.0003048
    offset_func = offset_coordinate_feet
    x_size_km = x_size * km_per_ft
    y_size_km = y_size * km_per_ft
  elif units == "m":
    offset_func = offset_coordinate_m
    km_per_m = 0.001
    x_size_km = x_size * km_per_m
    y_size_km = y_size * km_per_m
  else:
    raise ValueError(f"Unsupported units: {units}")

  for block in blocks:
    x = block["x"]
    y = block["y"]
    x_offset = x * x_size
    y_offset = y * y_size
    pt_y, pt_x = offset_func(o_y, o_x, y_offset, x_offset)
    geo = create_geo_rect_shape_km(pt_y, pt_x, x_size_km, y_size_km, "nw")
    block["geometry"] = geo

  return blocks


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