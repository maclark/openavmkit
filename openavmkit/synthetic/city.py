from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from shapely.geometry import LineString

import openavmkit
from openavmkit.utilities.geometry import offset_coordinate_feet, create_geo_rect, create_geo_rect_shape_km, \
  offset_coordinate_m, create_geo_rect_shape_deg


class SynCity:

  def __init__(self, params):
      self.latitude = params['latitude']
      self.longitude = params['longitude']
      self.crs = params['crs']
      self.anchor = params['anchor']
      self.units = params["units"]

      self.cbd_width_in_blocks = params['cbd_width_in_blocks']
      self.cbd_height_in_blocks = params['cbd_height_in_blocks']

      if self.units == "imperial":
        self.base_parcel_size = (150, 100)
      elif self.units == "metric":
        self.base_parcel_size = (45, 30)
      else:
        raise ValueError(f"Invalid units: {self.units}")

      self.width_in_blocks = params['width_in_blocks']
      self.height_in_blocks = params['height_in_blocks']

      self.max_width = self.base_parcel_size[0] * self.width_in_blocks
      self.max_height = self.base_parcel_size[1] * self.height_in_blocks

      self.gdf_roads:gpd.GeoDataFrame|None = None
      self.gdf_blocks:gpd.GeoDataFrame|None = None
      self.gdf_parcels:gpd.GeoDataFrame|None = None

      self.setup()


  def setup(self):
    # Set basic block size
    self.rows_per_block = 2
    self.parcels_per_row = 5

    self.block_size = (
      self.base_parcel_size[0] * self.rows_per_block,
      self.base_parcel_size[1] * self.parcels_per_row
    )

    self.width_in_blocks = self.width_in_blocks #int(self.max_width/self.block_size[0])
    self.height_in_blocks = self.height_in_blocks #int(self.max_height/self.block_size[1])
    print(f"Width: {self.width_in_blocks}, Height: {self.height_in_blocks}")

    self.build_grid()


  def build_grid(self):

    self.medium_road_every_n_blocks = 8
    self.large_road_every_n_blocks = 32
    self.sector_every_n_blocks = 48

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

    eighth_width = int(self.width_in_blocks / 8)
    eighth_height = int(self.height_in_blocks / 8)
    quarter_width = int(self.width_in_blocks / 4)
    quarter_height = int(self.height_in_blocks / 4)

    parcels_per_row = self.parcels_per_row
    rows_per_block = self.rows_per_block
    max_floors = 1

    # Append blocks
    for x in range(self.width_in_blocks):
      for y in range(self.height_in_blocks):
        in_cbd = False

        medium_roads = ((x % self.medium_road_every_n_blocks == 0) or ((x+1) % self.medium_road_every_n_blocks == 0)) + \
                       ((y % self.medium_road_every_n_blocks == 0) or ((y+1) % self.medium_road_every_n_blocks == 0))

        large_roads = ((x % self.large_road_every_n_blocks == 0) or ((x+1) % self.large_road_every_n_blocks == 0)) + \
                      ((y % self.large_road_every_n_blocks == 0) or ((y+1) % self.large_road_every_n_blocks == 0))

        road_type_n = 1
        road_type_s = 1
        road_type_w = 1
        road_type_e = 1

        on_medium_road = False
        on_large_road = False

        if x % self.medium_road_every_n_blocks == 0:
          road_type_w = 2
          on_medium_road = True
        if x % self.large_road_every_n_blocks == 0:
          road_type_w = 3
          on_large_road = True

        if (x+1) % self.medium_road_every_n_blocks == 0:
          road_type_e = 2
          on_medium_road = True
        if (x+1) % self.large_road_every_n_blocks == 0:
          road_type_e = 3
          on_large_road = True

        if y % self.medium_road_every_n_blocks == 0:
          road_type_s = 2
          on_medium_road = True
        if y % self.large_road_every_n_blocks == 0:
          road_type_s = 3
          on_large_road = True

        if (y+1) % self.medium_road_every_n_blocks == 0:
          road_type_n = 2
          on_medium_road = True
        if (y+1) % self.large_road_every_n_blocks == 0:
          road_type_n = 3
          on_large_road = True

        neighborhood_x = int(x/self.medium_road_every_n_blocks)
        neighborhood_y = int(y/self.medium_road_every_n_blocks)

        district_x = int(x/self.large_road_every_n_blocks)
        district_y = int(y/self.large_road_every_n_blocks)

        # get half the remainder between width in blocks and sector size
        sector_offset_x = int((self.width_in_blocks % self.sector_every_n_blocks))
        sector_offset_y = int((self.height_in_blocks % self.sector_every_n_blocks))

        sector_x = int((x+sector_offset_x)/(self.sector_every_n_blocks))
        sector_y = int((y+sector_offset_y)/(self.sector_every_n_blocks))

        neighborhood = f"{neighborhood_x}_x_{neighborhood_y}"
        district = f"{district_x}_x_{district_y}"
        sector = f"{sector_x}_x_{sector_y}"

        dist_to_cbd_x = abs(x - cbd_center_x)
        dist_to_cbd_y = abs(y - cbd_center_y)

        density = 1

        if dist_to_cbd_x < eighth_width and dist_to_cbd_y < eighth_height:
          density = 3
        elif dist_to_cbd_x < quarter_width and dist_to_cbd_y < quarter_height:
          density = 2

        if medium_roads < 1 and large_roads == 0:
          land_use = "R"
          density -= 1
        elif medium_roads < 2 and large_roads < 2:
          land_use = "M"
          density -= 1
        else:
          if on_large_road:
            land_use = "C"
          else:
            land_use = "M"

        if (x >= cbd_start_x and x < cbd_end_x) and (y >= cbd_start_y and y < cbd_end_y):
          in_cbd = True
          land_use = "CBD"
          density = 4
        if in_cbd is False and x >= cbd_end_x and y >= cbd_start_y and y < cbd_end_y:
          if land_use in ["R", "M", "C"]:
            land_use = "I"
            density += 1

        # adjacent to CBD
        if (x == cbd_start_x-1 or x == cbd_end_x) and (y == cbd_start_y-1 or y == cbd_end_y):
          if land_use == "R":
            land_use = "M"

        density = min(density, 4)

        zoning = _get_zoning_name(land_use, density, in_cbd)
        if in_cbd:
          neighborhood = f"CBD_{neighborhood}"
          district = f"CBD_{district}"
          sector = f"CBD"

        rows_per_block, parcels_per_row, max_floors, lot_coverage = _get_density_stats(density)

        block = {
          "x": x,
          "y": y,
          "cbd": in_cbd,
          "zoning": zoning,
          "land_use": land_use,
          "density": density,
          "neighborhood": neighborhood,
          "district": district,
          "sector": sector,
          "parcels_per_row": parcels_per_row,
          "rows_per_block": rows_per_block,
          "max_floors": max_floors,
          "lot_coverage": lot_coverage,
          "road_type_n": road_type_n,
          "road_type_s": road_type_s,
          "road_type_w": road_type_w,
          "road_type_e": road_type_e
        }
        blocks.append(block)

    # Append vertical rows
    for x in range(self.width_in_blocks+1):
      x_type = 1
      if x % self.medium_road_every_n_blocks == 0:
        x_type = 2
      if x % self.large_road_every_n_blocks == 0:
        x_type = 3

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
      y_type = 1
      if y % self.medium_road_every_n_blocks == 0:
        y_type = 2
      if y % self.large_road_every_n_blocks == 0:
        y_type = 3

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
    self.gdf_parcels = make_geo_parcels_from_city(self, blocks, units)


# Brazos county facts:
# - CBD block:
#   - 350x375 feet
#   - 5-10 parcels per block
# - Residential parcel:
#   - 150-160 x 50-80 feet
#   - 6-10 parcels per "strip"
#   - 2:3 aspect ratio



def _get_zoning_name(land_use: str, density: int, in_cbd: bool):
  zoning = f"{land_use}{density}"
  if in_cbd:
    zoning = "CBD"
  return zoning


def _get_density_stats(density: int)->tuple:

  if density == 0:
    rows_per_block  = 1
    parcels_per_row = 2
    max_floors      = 2
    lot_coverage    = 0.25
  elif density == 1:
    rows_per_block  = 1
    parcels_per_row = 4
    max_floors      = 2
    lot_coverage    = 0.50
  elif density == 2:
    rows_per_block  = 2
    parcels_per_row = 5
    max_floors      = 4
    lot_coverage    = 0.50
  elif density == 3:
    rows_per_block  = 2
    parcels_per_row = 8
    max_floors      = 6
    lot_coverage    = 0.75
  elif density == 4:
    rows_per_block  = 2
    parcels_per_row = 8
    max_floors      = 10
    lot_coverage    = 0.90
  else:
    raise ValueError(f"Invalid density: {density}")

  return rows_per_block, parcels_per_row, max_floors, lot_coverage


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


def make_geo_parcels_from_city(city: SynCity, blocks: list, units: str)->gpd.GeoDataFrame:
  o_y, o_x = _get_draw_anchor_coords(city)
  return make_geo_parcels(o_y, o_x, city.block_size[1], city.block_size[0], blocks, units, city.crs)


def make_geo_parcels(latitude, longitude, block_size_y, block_size_x, blocks: list, units: str, crs: Any)->gpd.GeoDataFrame:
  blocks = make_geo_blocks(latitude, longitude, block_size_y, block_size_x, blocks, units, crs)
  parcels = make_geo_parcels_raw(blocks, units, crs)
  return parcels


def make_geo_blocks_from_city(city: SynCity, blocks: list, units: str)->gpd.GeoDataFrame:
  o_y, o_x = _get_draw_anchor_coords(city)
  return make_geo_blocks(o_y, o_x, city.block_size[1], city.block_size[0], blocks, units, city.crs)


def make_geo_blocks(latitude, longitude, block_size_y, block_size_x, blocks: list, units: str, crs: Any)->gpd.GeoDataFrame:
  blocks = make_geo_blocks_raw(latitude, longitude, block_size_y, block_size_x, blocks, units)
  gdf = gpd.GeoDataFrame(data=blocks, geometry="geometry", crs=crs)
  return gdf


def make_geo_parcels_raw(blocks: gpd.GeoDataFrame, units: str, crs: Any)->gpd.GeoDataFrame|None:

  parcels = []

  # iterate through rows of blocks:
  for i, block in blocks.iterrows():
    # get the geometry of the block
    geo = block["geometry"]

    # get the upper left hand coordinate of the block's bounding box:
    origin_x, origin_y = geo.bounds[0], geo.bounds[3]
    max_width = geo.bounds[2] - geo.bounds[0]
    max_height = geo.bounds[3] - geo.bounds[1]

    rows_per_block = block["rows_per_block"]
    parcels_per_row = block["parcels_per_row"]

    parcel_width = max_width / rows_per_block
    parcel_height = max_height / parcels_per_row

    if i % 1000 == 0:
      perc = (i / len(blocks))
      print(f"{perc:6.2%} Building parcels for block {i} ({block['x']}, {block['y']})")

    print(f"Block: {i}, X: {block['x']}, Y: {block['y']}, rows_per_block = {rows_per_block}")

    for row in range(0, rows_per_block):
      for parcel in range(0, parcels_per_row):

        # get the coordinates of the parcel
        x = origin_x + (row * parcel_width)
        y = origin_y - (parcel * parcel_height)

        print(f"--> Parcel: {parcel}, Row: {row}, X: {x}, Y: {y}")


        # create a rectangle for the parcel
        rect = create_geo_rect_shape_deg(y, x, parcel_width, parcel_height, "nw")

        minx = rect.bounds[0]
        miny = rect.bounds[1]
        maxx = rect.bounds[2]
        maxy = rect.bounds[3]

        longitude = (minx + maxx) / 2
        latitude = (miny + maxy) / 2

        block_x = block["x"]
        block_y = block["y"]
        block_name = f"{block_x}_x_{block_y}"

        key = f"{block_x}-{block_y}-{row}-{parcel}"

        road_type_w = block["road_type_w"]
        road_type_e = block["road_type_e"]
        road_type_n = block["road_type_n"]
        road_type_s = block["road_type_s"]

        road_type = 0
        is_corner_lot = False
        corner_lot_type = 0
        road_type_ew = 0
        road_type_ns = 0

        if rows_per_block == 1:
          # both west & east side
          road_type_ew = max(road_type_w, road_type_e)
        else:
          if row == 0:
            # west side
            road_type_ew = max(road_type_w, road_type_ew)
          if row == rows_per_block-1:
            # east side
            road_type_ew = max(road_type_e, road_type_ew)

        if parcels_per_row == 1:
          # both north & south side
          road_type_ns = max(road_type_n, road_type_s)
        else:
          if parcel == 0:
            # north side
            road_type_ns = max(road_type_n, road_type_ns)
          if parcel == parcels_per_row-1:
            # south side
            road_type_ns = max(road_type_s, road_type_ns)

        road_type = max(road_type_ew, road_type_ns)

        print(f"--> road_type = {road_type}, road_type_ew = {road_type_ew}, road_type_ns = {road_type_ns}")
        print(f"----> w = {road_type_w}")
        print(f"----> e = {road_type_e}")

        if road_type_ew > 0 and road_type_ns > 0:
          is_corner_lot = True
          corner_lot_type = max(road_type_ew, road_type_ns)

        land_use = block["land_use"]
        density = block["density"]
        max_floors = block["max_floors"]
        lot_coverage = block["lot_coverage"]
        apartments_allowed = land_use in ["C", "CBD"]
        zoning = block["zoning"]

        # if it's on a major road, then some additional density is allowed
        if road_type == 3:
          density += 1
          density = min(density, 4)
          zoning = _get_zoning_name(land_use, density, block["cbd"])

          # density bump along streets only applies to floor limit and lot coverage
          _, _, max_floors, lot_coverage = _get_density_stats(density)

          # apartments in mixed-use zones only allowed along major roads
          if land_use in ["M"]:
            apartments_allowed = True

        parcel = {
          "key": key,
          "block": block_name,
          "zoning": zoning,
          "land_use": block["land_use"],
          "density": density,
          "neighborhood": block["neighborhood"],
          "district": block["district"],
          "sector": block["sector"],
          "max_floors": max_floors,
          "lot_coverage": lot_coverage,
          "apartments_allowed": apartments_allowed,
          "road_type": road_type,
          "is_corner_lot": is_corner_lot,
          "corner_lot_type": corner_lot_type,
          "latitude": latitude,
          "longitude": longitude,
          "geometry": rect
        }
        parcels.append(parcel)

  gdf = gpd.GeoDataFrame(data=parcels, geometry="geometry", crs=crs)

  crs = openavmkit.utilities.geometry.get_crs(gdf, "equal_area")
  gdf[f"area_land_sqm"] = gdf.to_crs(crs).geometry.area

  if units == "ft":
    gdf[f"area_land_sqft"] = gdf[f"area_land_sqm"] * 10.7639
    gdf.drop(columns=["area_land_sqm"], inplace=True)

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