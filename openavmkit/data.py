import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Dict, Any, Optional
from shapely.geometry import Point

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import geopandas as gpd
from pandas import Series
from scipy.spatial._ckdtree import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union
import warnings
import traceback

from openavmkit.calculations import _crawl_calc_dict_for_fields, perform_calculations, perform_tweaks
from openavmkit.filters import resolve_filter, select_filter
from openavmkit.utilities.assertions import dfs_are_equal
from openavmkit.utilities.cache import check_cache, write_cache, read_cache, get_cached_df, write_cached_df
from openavmkit.utilities.data import combine_dfs, div_field_z_safe, merge_and_stomp_dfs
from openavmkit.utilities.geometry import get_crs, clean_geometry, identify_irregular_parcels, get_exterior_coords, \
	geolocate_point_to_polygon, is_likely_epsg4326
from openavmkit.utilities.settings import get_fields_categorical, get_fields_impr, get_fields_boolean, \
  get_fields_numeric, get_model_group_ids, get_fields_date, get_long_distance_unit, get_valuation_date, get_center

from openavmkit.utilities.census import get_creds_from_env_census, init_service_census, match_to_census_blockgroups
from openavmkit.utilities.census import CensusService
from openavmkit.utilities.openstreetmap import init_service_openstreetmap
from openavmkit.utilities.overture import init_service_overture
from openavmkit.inference import get_inference_model, perform_spatial_inference
from openavmkit.utilities.timing import TimingData
from pyproj import CRS


@dataclass
class SalesUniversePair:
  """
  A container for the sales and universe DataFrames, many functions operate on this data structure. This data structure
  is necessary because the sales and universe DataFrames are often used together and need to be passed around together.
  The sales represent transactions and any known data at the time of the transaction, while the universe represents
  the current state of all parcels. The sales dataframe specifically allows for duplicate primary parcel keys, since an
  individual parcel may have sold multiple times. The universe dataframe should have no duplicate primary parcel keys.

  Attributes:
      sales (pd.DataFrame): DataFrame containing sales data.
      universe (pd.DataFrame): DataFrame containing universe (parcel) data.
  """
  sales: pd.DataFrame
  universe: pd.DataFrame

  def __getitem__(self, key):
    """
    Allow dictionary-like access to attributes.

    :param key: Attribute name, either "sales" or "universe".
    :type key: str
    :returns: The corresponding DataFrame.
    :rtype: pd.DataFrame
    """
    return getattr(self, key)


  def copy(self):
    """
    Create a copy of the SalesUniversePair object.

    :returns: A new SalesUniversePair object with copied DataFrames.
    :rtype: SalesUniversePair
    """
    return SalesUniversePair(self.sales.copy(), self.universe.copy())


  def set(self, key, value):
    """
    Set the sales or universe DataFrame.

    :param key: Either "sales" or "universe".
    :type key: str
    :param value: The new DataFrame.
    :type value: pd.DataFrame
    :raises ValueError: If an invalid key is provided.
    """
    if key == "sales":
      self.sales = value
    elif key == "universe":
      self.universe = value
    else:
      raise ValueError(f"Invalid key: {key}")

  def update_sales(self, new_sales: pd.DataFrame, allow_remove_rows: bool = False):
    """
    Update the sales DataFrame with new information as an overlay without redundancy.

    This function lets you push updates to "sales" while keeping it as an "overlay" that doesn't contain any redundant information.

    - First we note what fields were in sales last time.
    - Then we note what sales are in universe but were not in sales.
    - Finally, we determine the new fields generated in new_sales that are not in the previous sales or in the universe.
    - A modified version of df_sales is created with only two changes:
      - Reduced to the correct selection of keys.
      - Addition of the newly generated fields.

    :param new_sales: New sales DataFrame with updates.
    :type new_sales: pd.DataFrame
    :param allow_remove_rows: If True, allows the update to remove rows from sales. If False, preserves all original rows.
    :type allow_remove_rows: bool
    :returns: None
    """
    old_fields = self.sales.columns.values
    univ_fields = [field for field in self.universe.columns.values if field not in old_fields]
    new_fields = [field for field in new_sales.columns.values if field not in old_fields and field not in univ_fields]
    return_keys = new_sales["key_sale"].values
    reconciled = new_sales.copy()
    reconciled = reconciled[reconciled["key_sale"].isin(return_keys)]
    reconciled = combine_dfs(reconciled, new_sales[["key_sale"] + new_fields], index="key_sale")

    old_sales = self.sales.copy()
    return_keys = new_sales["key_sale"].values
    if not allow_remove_rows and len(return_keys) > len(old_sales):
      raise ValueError("The new sales DataFrame contains more keys than the old sales DataFrame. update_sales() may only be used to shrink the dataframe or keep it the same size. Use set() if you intend to replace the sales dataframe.")

    if allow_remove_rows:
      old_sales = old_sales[old_sales["key_sale"].isin(return_keys)].reset_index(drop=True)
    reconciled = combine_dfs(old_sales, new_sales[["key_sale"] + new_fields].copy().reset_index(drop=True), index="key_sale")
    self.sales = reconciled


SUPKey = Literal["sales", "universe"]


def get_hydrated_sales_from_sup(sup: SalesUniversePair):
  """
  Merge the sales and universe DataFrames to "hydrate" the sales data. The sales data represents transactions and any
  known data at the time of the transaction, while the universe data represents the current state of all parcels. When
  we merge the two sets, the sales data overrides any existing data in the universe data. This is useful for creating
  a "hydrated" sales DataFrame that contains all the information available at the time of the sale (it is assumed that
  any difference between the current state of the parcel and the state at the time of the sale is accounted for in the
  sales data).

  If the merged DataFrame contains a "geometry" column and the original sales did not,
  the result is converted to a GeoDataFrame.

  :param sup: SalesUniversePair containing sales and universe DataFrames.
  :type sup: SalesUniversePair
  :returns: The merged (hydrated) sales DataFrame.
  :rtype: pd.DataFrame or gpd.GeoDataFrame
  """
  df_sales = sup["sales"]
  df_univ = sup["universe"].copy()
  df_univ = df_univ[df_univ["key"].isin(df_sales["key"].values)].reset_index(drop=True)
  df_merged = merge_and_stomp_dfs(df_sales, df_univ, df2_stomps=False)

  if "geometry" in df_merged and "geometry" not in df_sales:
    # convert df_merged to geodataframe:
    df_merged = gpd.GeoDataFrame(df_merged, geometry="geometry")

  return df_merged


def enrich_time(df: pd.DataFrame, time_formats: dict, settings: dict) -> pd.DataFrame:
  """
  Enrich the DataFrame by converting specified time fields to datetime and deriving additional fields.

  For each key in time_formats, converts the column to datetime. Then, if a field with the prefix "sale" exists,
  enriches the dataframe with additional time fields (e.g., "sale_year", "sale_month", "sale_age_days").

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param time_formats: Dictionary mapping field names to datetime formats.
  :type time_formats: dict
  :param settings: Settings dictionary.
  :type settings: dict
  :returns: DataFrame with enriched time fields.
  :rtype: pandas.DataFrame
  """
  for key in time_formats:
    time_format = time_formats[key]
    if key in df:
      df[key] = pd.to_datetime(df[key], format=time_format, errors="coerce")

  for prefix in ["sale"]:
    do_enrich = False
    for col in df.columns.values:
      if f"{prefix}_" in col:
        do_enrich = True
        break
    if do_enrich:
      df = _enrich_time_field(df, prefix, add_year_month=True, add_year_quarter=True)
      if prefix == "sale":
        df = _enrich_sale_age_days(df, settings)

  return df


def simulate_removed_buildings(df: pd.DataFrame, settings: dict, idx_vacant: Series = None):
  """
  Simulate removed buildings by changing improvement fields to values that reflect the absence of a building.

  For all improvement fields, fills categorical fields with "UNKNOWN", numeric fields with 0, and boolean fields with
  False for the rows specified by idx_vacant (or all rows if idx_vacant is None).

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :param idx_vacant: Optional Series indicating which rows are vacant.
  :type idx_vacant: pandas.Series, optional
  :returns: Updated DataFrame.
  :rtype: pandas.DataFrame
  """
  if idx_vacant is None:
    # do the whole thing:
    idx_vacant = df.index

  fields_impr = get_fields_impr(settings, df)

  # fill unknown values for categorical improvements:
  fields_impr_cat = fields_impr["categorical"]
  fields_impr_num = fields_impr["numeric"]
  fields_impr_bool = fields_impr["boolean"]

  for field in fields_impr_cat:
    df.loc[idx_vacant, field] = "UNKNOWN"

  for field in fields_impr_num:
    df.loc[idx_vacant, field] = 0

  for field in fields_impr_bool:
    df.loc[idx_vacant, field] = False

  # just to be safe, ensure that the "bldg_area_finished_sqft" field is set to 0 for vacant sales
  # and update "is_vacant" to perfectly match
  # TODO: if we add support for a user having a custom vacancy filter, we will need to adjust this
  if "bldg_area_finished_sqft" in df:
    df.loc[idx_vacant, "bldg_area_finished_sqft"] = 0
    df["is_vacant"] = False
    df.loc[idx_vacant, "is_vacant"] = True

  return df


def get_sale_field(settings: dict, df:pd.DataFrame=None) -> str:
  """
  Determine the appropriate sale price field ("sale_price" or "sale_price_time_adj") based on time adjustment settings.

  :param settings: Settings dictionary.
  :type settings: dict
  :param df: Optional DataFrame to check field existence.
  :type df: pandas.DataFrame, optional
  :returns: Field name to be used for sale price.
  :rtype: str
  """
  ta = settings.get("modeling", {}).get("instructions", {}).get("time_adjustment", {})
  use = ta.get("use", True)
  if use:
    return "sale_price_time_adj"
  if df is not None:
    if "sale_price_time_adj" in df:
      return "sale_price_time_adj"
  return "sale_price"


def get_vacant_sales(df_in: pd.DataFrame, settings: dict, invert: bool = False) -> pd.DataFrame:
  """
  Filter the sales DataFrame to return only vacant (unimproved) sales.

  :param df_in: Input DataFrame.
  :type df_in: pandas.DataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :param invert: If True, return non-vacant (improved) sales.
  :type invert: bool, optional
  :returns: Filtered DataFrame containing (or excluding) vacant sales.
  :rtype: pandas.DataFrame
  """
  df = df_in.copy()
  df = _boolify_column_in_df(df, "vacant_sale")
  idx_vacant_sale = df["vacant_sale"].eq(True)
  if invert:
    idx_vacant_sale = ~idx_vacant_sale
  df_vacant_sales = df[idx_vacant_sale].copy()
  return df_vacant_sales


def is_series_all_bools(series: pd.Series) -> bool:
  dtype = series.dtype
  if dtype == bool:
    return True
  # check all unique values:
  uniques = series.unique()
  for unique in uniques:
    if type(unique) != bool:
      return False
  return True


def get_vacant(df_in: pd.DataFrame, settings: dict, invert: bool = False) -> pd.DataFrame:
  """
  Filter the DataFrame based on the 'is_vacant' column.

  :param df_in: Input DataFrame.
  :type df_in: pandas.DataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :param invert: If True, return non-vacant rows.
  :type invert: bool, optional
  :returns: DataFrame filtered by the 'is_vacant' flag.
  :rtype: pandas.DataFrame
  :raises ValueError: If 'is_vacant' column is not boolean.
  """
  df = df_in.copy()
  is_vacant_dtype = df["is_vacant"].dtype
  if is_vacant_dtype != bool:
    raise ValueError(f"The 'is_vacant' column must be a boolean type (found: {is_vacant_dtype})")
  idx_vacant = df["is_vacant"].eq(True)
  if invert:
    idx_vacant = ~idx_vacant
  df_vacant = df[idx_vacant].copy()
  return df_vacant


def get_sales(df_in: pd.DataFrame, settings: dict, vacant_only: bool = False, df_univ: pd.DataFrame = None) -> pd.DataFrame:
  """
  Retrieve valid sales from the input DataFrame. Also simulates removed buildings if applicable.

  Filters for sales with a positive sale price, valid_sale marked True.
  If vacant_only is True, only includes rows where vacant_sale is True.

  :param df_in: Input DataFrame containing sales.
  :type df_in: pandas.DataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :param vacant_only: If True, return only vacant sales.
  :type vacant_only: bool, optional
  :returns: Filtered DataFrame of valid sales.
  :rtype: pandas.DataFrame
  :raises ValueError: If required boolean columns are not of boolean type.
  """
  df = df_in.copy()
  valid_sale_dtype = df["valid_sale"].dtype
  if valid_sale_dtype != bool:
    if is_series_all_bools(df["valid_sale"]):
      df["valid_sale"] = df["valid_sale"].astype(bool)
    else:
      raise ValueError(f"The 'valid_sale' column must be a boolean type (found: {valid_sale_dtype}) with values: {df['valid_sale'].unique()}")

  if "vacant_sale" in df:
    vacant_sale_dtype = df["vacant_sale"].dtype
    if vacant_sale_dtype != bool:
      if is_series_all_bools(df["vacant_sale"]):
        df["vacant_sale"] = df["vacant_sale"].astype(bool)
      else:
        raise ValueError(f"The 'vacant_sale' column must be a boolean type (found: {vacant_sale_dtype}) with values: {df['vacant_sale'].unique()}")
    # check for vacant sales:
    idx_vacant_sale = df["vacant_sale"].eq(True)

    # simulate removed buildings for vacant sales
    # (if we KNOW it was a vacant sale, then the building characteristics have to go)
    df = simulate_removed_buildings(df, settings, idx_vacant_sale)

    # TODO: smell
    if "is_vacant" not in df and df_univ is not None:
      df = df.merge(df_univ[["key", "is_vacant"]], on="key", how="left")

    if "model_group" not in df and df_univ is not None:
      df = df.merge(df_univ[["key", "model_group"]], on="key", how="left")

    # if a property was NOT vacant at time of sale, but is vacant now, then the sale is invalid:
    idx_is_vacant = df["is_vacant"].eq(True)
    df.loc[~idx_vacant_sale & idx_is_vacant, "valid_sale"] = False
  idx_sale_price = df["sale_price"].gt(0)
  idx_valid_sale = df["valid_sale"].eq(True)
  idx_is_vacant = df["vacant_sale"].eq(True)
  idx_all = idx_sale_price & idx_valid_sale & (idx_is_vacant if vacant_only else True)


  df_sales: pd.DataFrame = df[
    df["sale_price"].gt(0) &
    df["valid_sale"].eq(True) &
    (df["vacant_sale"].eq(True) if vacant_only else True)
  ].copy()

  return df_sales


def get_report_locations(settings: dict, df: pd.DataFrame = None) -> list[str]:
  """
  Retrieve report location fields from settings. These are location fields that will be used in report breakdowns, such
  as for ratio studies.

  :param settings: Settings dictionary.
  :type settings: dict
  :param df: Optional DataFrame to filter available locations.
  :type df: pandas.DataFrame, optional
  :returns: List of report location field names.
  :rtype: list[str]
  """
  locations = settings.get("field_classification", {}).get("important", {}).get("report_locations", [])
  if df is not None:
    locations = [loc for loc in locations if loc in df]
  return locations


def get_locations(settings: dict, df: pd.DataFrame = None) -> list[str]:
  """
  Retrieve location fields from settings. These are all the fields that are considered locations.

  :param settings: Settings dictionary.
  :type settings: dict
  :param df: Optional DataFrame to filter available locations.
  :type df: pandas.DataFrame, optional
  :returns: List of location field names.
  :rtype: list[str]
  """
  locations = settings.get("field_classification", {}).get("important", {}).get("locations", [])
  if df is not None:
    locations = [loc for loc in locations if loc in df]
  return locations


def get_important_fields(settings: dict, df: pd.DataFrame = None) -> list[str]:
  """
  Retrieve important field names from settings.

  :param settings: Settings dictionary.
  :type settings: dict
  :param df: Optional DataFrame to filter fields.
  :type df: pandas.DataFrame, optional
  :returns: List of important field names.
  :rtype: list[str]
  """
  imp = settings.get("field_classification", {}).get("important", {})
  fields = imp.get("fields", {})
  list_fields = []
  if df is not None:
    for field in fields:
      other_name = fields[field]
      if other_name in df:
        list_fields.append(other_name)
  return list_fields


def get_important_field(settings: dict, field_name: str, df: pd.DataFrame = None) -> str | None:
  """
  Retrieve the important field name for a given field alias from settings. For instance if you are using school district
  as your market area, you would look up "loc_market_area", which should be set to "school_district" in your settings.

  :param settings: Settings dictionary.
  :type settings: dict
  :param field_name: Identifier for the field.
  :type field_name: str
  :param df: Optional DataFrame to check field existence.
  :type df: pandas.DataFrame, optional
  :returns: The mapped field name if found, else None.
  :rtype: str or None
  """
  imp = settings.get("field_classification", {}).get("important", {})
  other_name = imp.get("fields", {}).get(field_name, None)
  if df is not None:
    if other_name is not None and other_name in df:
      return other_name
    else:
      return None
  return other_name


def get_field_classifications(settings: dict):
  """
  Retrieve a mapping of field names to their classifications (land, improvement or other) as well as their types
  (numeric, categorical, or boolean).

  :param settings: Settings dictionary.
  :type settings: dict
  :returns: Dictionary mapping field names to type and class.
  :rtype: dict
  """
  field_map = {}
  for ftype in ["land", "impr", "other"]:
    nums = get_fields_numeric(settings, None, False, [ftype])
    cats = get_fields_categorical(settings, None, False, [ftype])
    bools = get_fields_boolean(settings, None, [ftype])
    for field in nums:
      field_map[field] = {"type": ftype, "class": "numeric"}
    for field in cats:
      field_map[field] = {"type": ftype, "class": "categorical"}
    for field in bools:
      field_map[field] = {"type": ftype, "class": "boolean"}
  return field_map


def get_dtypes_from_settings(settings: dict):
  """
  Generate a dictionary mapping fields to their designated data types based on settings.

  :param settings: Settings dictionary.
  :type settings: dict
  :returns: Dictionary of field names to data type strings.
  :rtype: dict
  """
  cats = get_fields_categorical(settings, include_boolean=False)
  bools = get_fields_boolean(settings)
  nums = get_fields_numeric(settings, include_boolean=False)
  dtypes = {}
  for c in cats:
    dtypes[c] = "string"
  for b in bools:
    dtypes[b] = "bool"
  for n in nums:
    dtypes[n] = "Float64"
  return dtypes


def process_data(dataframes: dict[str, pd.DataFrame], settings: dict, verbose: bool = False) -> SalesUniversePair:
  """
  Process raw dataframes according to settings and return a SalesUniversePair.

  :param dataframes: Dictionary mapping keys to DataFrames.
  :type dataframes: dict[str, pd.DataFrame]
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints progress information.
  :type verbose: bool, optional
  :returns: A SalesUniversePair containing processed sales and universe data.
  :rtype: SalesUniversePair
  :raises ValueError: If required merge instructions or columns are missing.
  """
  s_data = settings.get("data", {})
  s_process = s_data.get("process", {})
  s_merge = s_process.get("merge", {})

  merge_univ: list | None = s_merge.get("universe", None)
  merge_sales: list | None = s_merge.get("sales", None)

  if merge_univ is None:
    raise ValueError("No \"universe\" merge instructions found. data.process.merge must have exactly two keys: \"universe\", and \"sales\"")
  if merge_sales is None:
    raise ValueError("No \"sales\" merge instructions found. data.process.merge must have exactly two keys: \"universe\", and \"sales\"")

  df_univ = _merge_dict_of_dfs(dataframes, merge_univ, settings, required_key="key")
  df_sales = _merge_dict_of_dfs(dataframes, merge_sales, settings, required_key="key_sale")

  if "valid_sale" not in df_sales:
    raise ValueError("The 'valid_sale' column is required in the sales data.")
  if "vacant_sale" not in df_sales:
    raise ValueError("The 'vacant_sale' column is required in the sales data.")
  # Print number and percentage of valid sales
  valid_count = df_sales["valid_sale"].sum()
  total_count = len(df_sales)
  valid_percent = (valid_count / total_count * 100) if total_count > 0 else 0
  print(f"Valid sales: {valid_count} ({valid_percent:.1f}% of {total_count} total)")
  df_sales = df_sales[df_sales["valid_sale"].eq(True)].copy().reset_index(drop=True)

  sup: SalesUniversePair = SalesUniversePair(universe=df_univ, sales=df_sales)

  sup = enrich_data(sup, s_process.get("enrich", {}), dataframes, settings, verbose=verbose)

  dupe_univ: dict|None = s_process.get("dupes", {}).get("universe", None)
  dupe_sales: dict|None = s_process.get("dupes", {}).get("sales", None)
  if dupe_univ:
    sup.set("universe", _handle_duplicated_rows(sup.universe, dupe_univ, verbose=verbose))
  if dupe_sales:
    sup.set("sales", _handle_duplicated_rows(sup.sales, dupe_sales, verbose=verbose))

  return sup


def enrich_data(sup: SalesUniversePair, s_enrich: dict, dataframes: dict[str, pd.DataFrame], settings: dict, verbose: bool = False) -> SalesUniversePair:
  """
  Enrich both sales and universe data based on enrichment instructions.

  Applies enrichment operations (e.g., spatial and basic enrichment) to both "sales" and "universe" DataFrames.

  :param sup: SalesUniversePair containing sales and universe data.
  :type sup: SalesUniversePair
  :param s_enrich: Enrichment instructions.
  :type s_enrich: dict
  :param dataframes: Dictionary of additional DataFrames.
  :type dataframes: dict[str, pd.DataFrame]
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints progress information.
  :type verbose: bool, optional
  :returns: Enriched SalesUniversePair.
  :rtype: SalesUniversePair
  """
  supkeys: list[SUPKey] = ["universe", "sales"]

  # Add the "both" entries to both "universe" and "sales" and delete the "both" entry afterward.
  if "both" in s_enrich:
    s_enrich2 = s_enrich.copy()
    s_both = s_enrich.get("both")
    for key in s_both:
      for supkey in supkeys:
        sup_entry = s_enrich.get(supkey, {})
        if key in sup_entry:
          # Check if the key already exists on "sales" or "universe"
          raise ValueError(f"Cannot enrich '{key}' twice -- found in both \"both\" and \"{supkey}\". Please remove one.")
        entry = s_both[key]
        # add the entry from "both" to both the "sales" & "universe" entry
        sup_entry2 = s_enrich2.get(supkey, {})
        sup_entry2[key] = entry
        s_enrich2[supkey] = sup_entry2
    del s_enrich2["both"]  # remove the now-redundant "both" key
    s_enrich = s_enrich2

  for supkey in supkeys:
    if verbose:
      print(f"Enriching {supkey}...")

    df = sup.sales if supkey == "sales" else sup.universe

    s_enrich_local: dict | None = s_enrich.get(supkey, None)

    if s_enrich_local is not None:
      # Handle Census enrichment for universe if enabled
      if supkey == "universe":
        df = _basic_geo_enrichment(df, settings, verbose=verbose)

        if "census" in s_enrich_local:
          df = _enrich_df_census(df, s_enrich_local.get("census", {}), verbose=verbose)
        if "openstreetmap" in s_enrich_local:
          df = _enrich_df_openstreetmap(df, s_enrich_local.get("openstreetmap", {}), s_enrich_local, dataframes, verbose=verbose, use_cache = True)

        df = _enrich_df_geometry(df, s_enrich_local, dataframes, settings, verbose=verbose)

      df = _enrich_df_basic(df, s_enrich_local, dataframes, settings, supkey == "sales", verbose=verbose)

      # if supkey == "universe":
      #   # enrich universe spatial lag fields
      #   df = _enrich_universe_spatial_lag(df, settings, verbose=verbose)

    # stuff to enrich whether the user has settings or not
    df = _enrich_vacant(df, settings)

    sup.set(supkey, df)

  return sup

def _enrich_df_census(df_in: pd.DataFrame | gpd.GeoDataFrame, census_settings: dict, verbose: bool = False) -> pd.DataFrame | gpd.GeoDataFrame:
  """
  Enrich a DataFrame with Census data by performing a spatial join with Census block groups.

  :param df: Input DataFrame or GeoDataFrame to enrich with Census data.
  :type df: pd.DataFrame | gpd.GeoDataFrame
  :param census_settings: Census enrichment settings.
  :type census_settings: dict
  :param verbose: If True, prints progress information.
  :type verbose: bool, optional
  :returns: DataFrame enriched with Census data.
  :rtype: pd.DataFrame | gpd.GeoDataFrame
  """
  if not census_settings.get("enabled", False):
    return df_in

  if verbose:
    print("Enriching with Census data...")

  df_out = get_cached_df(df_in, "census", "key", census_settings)
  if df_out is not None:
    if verbose:
      print("--> found cached data")
    return df_out

  df = df_in.copy()

  try:
    # Get Census credentials and initialize service
    creds = get_creds_from_env_census()
    census_service = init_service_census(creds)

    # Get FIPS code from settings
    fips_code = census_settings.get("fips", "")
    if not fips_code:
      warnings.warn("Census enrichment enabled but no FIPS code provided in settings")
      return df

    year = census_settings.get("year", 2022)
    if verbose:
      print("Getting Census Data...")

    # Get Census data with boundaries
    census_data, census_boundaries = census_service.get_census_data_with_boundaries(
      fips_code=fips_code,
      year=year
    )

    # Spatial join with universe data only
    if not isinstance(df, gpd.GeoDataFrame):
      warnings.warn("DataFrame is not a GeoDataFrame, skipping Census enrichment")
      return df

    # Get census columns to keep
    census_cols_to_keep = ['std_geoid', 'median_income', 'total_pop']

    # Ensure all census columns exist in the census_boundaries
    missing_cols = [col for col in census_cols_to_keep if col not in census_boundaries.columns]
    if missing_cols:
      # Filter to only include columns that exist
      census_cols_to_keep = [col for col in census_cols_to_keep if col in census_boundaries.columns]

    # Create a copy of census_boundaries with only the columns we need
    census_boundaries_subset = census_boundaries[['geometry'] + census_cols_to_keep].copy()

    if verbose:
      print("Performing spatial join with Census Data...")

    # Perform the spatial join
    df = match_to_census_blockgroups(
      gdf=df,
      census_gdf=census_boundaries_subset,
      join_type="left"
    )

    write_cached_df(df_in, df, "census", "key", census_settings)

    return df

  except Exception as e:
    warnings.warn(f"Failed to enrich with Census data: {str(e)}")
    return df

def _enrich_df_openstreetmap(df_in: pd.DataFrame | gpd.GeoDataFrame, osm_settings: dict, s_enrich_this: dict, dataframes: dict, verbose: bool = False, use_cache: bool = False) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Enrich a DataFrame with OpenStreetMap data.
    
    Args:
        df (pd.DataFrame | gpd.GeoDataFrame): DataFrame to enrich
        osm_settings (dict): Settings for OpenStreetMap enrichment
        s_enrich_this (dict): Enrichment settings to update with distances configuration
        dataframes (dict): Dictionary of all dataframes, will be updated with OSM features
        verbose (bool): Whether to print detailed information
        use_cache (bool): Whether to use cached data if available
        
    Returns:
        pd.DataFrame | gpd.GeoDataFrame: DataFrame enriched with OpenStreetMap data
    """

    df = df_in.copy()

    if use_cache:
      print("Checking cache for OpenStreetMap data...")
      df_out = get_cached_df(df, "osm/all", "key", osm_settings)
      if df_out is not None:
        # Load cached features into dataframes dictionary
        for feature in ['water_bodies', 'transportation', 'educational', 'parks', 'golf_courses']:
          feature_cache = get_cached_df(df, f"osm/{feature}", "key", osm_settings)
          if feature_cache is not None:
            dataframes[feature] = feature_cache
            # Also check for top features
            top_feature = f"{feature}_top"
            top_cache = get_cached_df(df, f"osm/{top_feature}", "key", osm_settings)
            if top_cache is not None:
              dataframes[top_feature] = top_cache
        return df_out

    try:
        if not osm_settings.get('enabled', False):
            if verbose:
                print("OpenStreetMap enrichment disabled, skipping all OSM features")
            return df
            
        # Initialize OpenStreetMap service
        osm_service = init_service_openstreetmap(osm_settings)
        
        # Convert DataFrame to GeoDataFrame if it isn't already
        if not isinstance(df, gpd.GeoDataFrame):
            warnings.warn("DataFrame is not a GeoDataFrame, skipping OpenStreetMap enrichment")
            return df
            
        # Ensure the GeoDataFrame is in WGS84 (EPSG:4326) before getting bounds
        original_crs = df.crs

        if original_crs is None:
            warnings.warn("GeoDataFrame has no CRS set, attempting to infer EPSG:4326")
            if is_likely_epsg4326(df):
                df.set_crs(epsg=4326, inplace=True)
            else:
                raise ValueError("Cannot determine CRS of input GeoDataFrame")
        elif not original_crs.equals(CRS.from_epsg(4326)):
            df = df.to_crs(epsg=4326)
            
        # Calculate the bounding box by unioning all geometries first
        # This ensures we get the full extent of all geometries
        all_geoms = df.geometry.unary_union
        bbox = all_geoms.bounds
        # Process each feature based on settings

        # Define a dictionary to hold feature configurations
        features_config = {
          "water_bodies": {
            "getter": osm_service.get_water_bodies,
            "verbose_label": "water bodies",
            "store_top": True,
            "error_method": "print",  # print error message with traceback
          },
          "transportation": {
            "getter": osm_service.get_transportation,
            "verbose_label": "transportation networks",
            "store_top": False,       # no top features for transportation
            "error_method": "warn",   # use warnings.warn
          },
          "educational": {
            "getter": osm_service.get_educational_institutions,
            "verbose_label": "educational institutions",
            "store_top": True,
            "error_method": "warn",
          },
          "parks": {
            "getter": osm_service.get_parks,
            "verbose_label": "parks",
            "store_top": True,
            "error_method": "warn",
          },
          "golf_courses": {
            "getter": osm_service.get_golf_courses,
            "verbose_label": "golf courses",
            "store_top": True,
            "error_method": "warn",
          },
        }

        # Loop through each feature configuration:
        for feature, config in features_config.items():
          # Check if feature is enabled in the osm_settings
          if osm_settings.get(feature, {}).get('enabled', False):
            if verbose:
              print(f"--> Getting {config['verbose_label']}...")
            try:
              # Call the designated getter function
              result = config["getter"](
                bbox=bbox,
                settings=osm_settings[feature],
                use_cache=use_cache
              )
              if verbose:
                if result.empty:
                  print(f"    No {config['verbose_label']} found")
                else:
                  print(f"--> Found {len(result)} {config['verbose_label']}")
              if not result.empty:
                # Save the primary result
                dataframes[feature] = result
                # Save top features if applicable
                if config["store_top"]:
                  top_key = feature + "_top"
                  if top_key in osm_service.features:
                    dataframes[top_key] = osm_service.features[top_key]
            except Exception as e:
              err_msg = f"Failed to get {config['verbose_label']}: {str(e)}"
              if config["error_method"] == "warn":
                warnings.warn(err_msg)
              else:
                print("ERROR " + err_msg)
                print("Traceback: " + traceback.format_exc())
        
        # Configure distance calculations to match settings file format
        distances = []
        
        # Get the distances configuration from settings
        distances_config = s_enrich_this.get('distances', [])
        distances_by_id = {d.get('id'): d for d in distances_config if isinstance(d, dict)}
        
        for feature_name in ['water_bodies', 'transportation', 'educational', 'parks', 'golf_courses']:
            if feature_name in dataframes:
                # Add base feature with its settings from the config
                if feature_name in distances_by_id:
                    distances.append(distances_by_id[feature_name])
                
                # Add top features for individual distances if available
                top_feature_name = f"{feature_name}_top"
                if top_feature_name in dataframes and top_feature_name in distances_by_id:
                    distances.append(distances_by_id[top_feature_name])
        
        # Add the distances configuration to the enrichment settings
        if distances:
            s_enrich_this['distances'] = distances

        write_cached_df(df_in, df, "osm/all", "key", osm_settings)

        return df
        
    except Exception as e:
        warnings.warn(f"Failed to enrich with OpenStreetMap data: {str(e)}")
        return df


def identify_parcels_with_holes(df: gpd.GeoDataFrame) -> (gpd.GeoDataFrame, gpd.GeoDataFrame):
  """
  Identify parcels with holes (interior rings) in their geometries.

  :param df: GeoDataFrame with parcel geometries.
  :type df: geopandas.GeoDataFrame
  :returns: GeoDataFrame with parcels containing interior rings.
  :rtype: geopandas.GeoDataFrame
  """
  # Identify parcels with holes
  def has_holes(geom):
    if geom.is_valid:
      if geom.geom_type == "Polygon":
        return len(geom.interiors) > 0
      elif geom.geom_type == "MultiPolygon":
        return any(len(p.interiors) > 0 for p in geom.geoms)
    return False

  parcels_with_holes = df[df.geometry.apply(has_holes)]
  # Remove duplicates:
  parcels_with_holes = parcels_with_holes.drop_duplicates(subset="key")
  return parcels_with_holes


# Private functions below:

def _enrich_sale_age_days(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
  """
  Enrich the DataFrame with a 'sale_age_days' column indicating the age in days since sale.

  :param df: Input DataFrame with a "sale_date" column.
  :type df: pandas.DataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :returns: DataFrame with an added "sale_age_days" column.
  :rtype: pandas.DataFrame
  """
  val_date = get_valuation_date(settings)
  # create a new field with dtype Int64
  df["sale_age_days"] = None
  df["sale_age_days"] = df["sale_age_days"].astype("Int64")
  sale_date_as_datetime = pd.to_datetime(df["sale_date"], format="%Y-%m-%d", errors="coerce")
  df.loc[~sale_date_as_datetime.isna(), "sale_age_days"] = (val_date - sale_date_as_datetime).dt.days
  return df


def _enrich_year_built(df: pd.DataFrame, settings: dict, is_sales: bool = False):
  """
  Enrich the DataFrame with building age information based on year built.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :param is_sales: Flag indicating if the DataFrame is sales data.
  :type is_sales: bool, optional
  :returns: DataFrame with new age fields.
  :rtype: pandas.DataFrame
  """
  val_date = get_valuation_date(settings)
  for prefix in ["bldg", "bldg_effective"]:
    col = f"{prefix}_year_built"
    if col in df:
      new_col = f"{prefix}_age_years"
      df = _do_enrich_year_built(df, col, new_col, val_date, is_sales)
  return df


def _do_enrich_year_built(df: pd.DataFrame, col: str, new_col: str, val_date: datetime, is_sales: bool = False) -> pd.DataFrame:
  """
  Calculate building age and add it as a new column.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param col: Column name for year built.
  :type col: str
  :param new_col: New column name for calculated age.
  :type new_col: str
  :param val_date: Valuation date.
  :type val_date: datetime
  :param is_sales: Flag indicating if processing sales data.
  :type is_sales: bool, optional
  :returns: DataFrame with the new age column.
  :rtype: pandas.DataFrame
  """
  if not is_sales:
    val_year = val_date.year
    df[new_col] = val_year - df[col]

    # Avoid 2000+ year old buildings whose year built is 0
    df.loc[
      df[col].isna() |
      df[col].le(0),
      new_col
    ] = 0
  else:
    df.loc[df["sale_year"].notna(), new_col] = df["sale_year"] - df[col]

    df.loc[
      df["sale_year"].isna() |
      df["sale_year"].le(0),
      new_col
    ] = 0
  return df


def _enrich_time_field(df: pd.DataFrame, prefix: str, add_year_month: bool = True, add_year_quarter: bool = True) -> pd.DataFrame:
  """
  Enrich a DataFrame with time-related fields based on a prefix.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param prefix: Prefix for time fields (e.g., "sale").
  :type prefix: str
  :param add_year_month: Whether to add a "year_month" field.
  :type add_year_month: bool, optional
  :param add_year_quarter: Whether to add a "year_quarter" field.
  :type add_year_quarter: bool, optional
  :returns: DataFrame with enriched time fields.
  :rtype: pandas.DataFrame
  :raises ValueError: If required date information is missing.
  """
  if f"{prefix}_date" not in df:
    # Check if we have _year, _month, and _day:
    if f"{prefix}_year" in df and f"{prefix}_month" in df and f"{prefix}_day" in df:
      date_str_series = (
          df[f"{prefix}_year"].astype(str).str.pad(4, fillchar="0") + "-" +
          df[f"{prefix}_month"].astype(str).str.pad(2, fillchar="0") + "-" +
          df[f"{prefix}_day"].astype(str).str.pad(2, fillchar="0")
      )
      df[f"{prefix}_date"] = pd.to_datetime(date_str_series, format="%Y-%m-%d", errors="coerce")
    else:
      raise ValueError(f"The dataframe does not contain a '{prefix}_date' column.")
  df[f"{prefix}_date"] = pd.to_datetime(df[f"{prefix}_date"], format="%Y-%m-%d", errors="coerce")
  df[f"{prefix}_year"] = df[f"{prefix}_date"].dt.year
  df[f"{prefix}_month"] = df[f"{prefix}_date"].dt.month
  df[f"{prefix}_quarter"] = df[f"{prefix}_date"].dt.quarter
  if add_year_month:
    df[f"{prefix}_year_month"] = df[f"{prefix}_date"].dt.to_period("M").astype("str")
  if add_year_quarter:
    df[f"{prefix}_year_quarter"] = df[f"{prefix}_date"].dt.to_period("Q").astype("str")
  checks = ["_year", "_month", "_day", "_year_month", "_year_quarter"]
  for check in checks:
    if f"{prefix}{check}" in df:
      if f"{prefix}_date" in df:
        if check in ["_year", "_month", "_day"]:
          date_value = None
          if check == "_year":
            date_value = df[f"{prefix}_date"].dt.year.astype("Int64")
          elif check == "_month":
            date_value = df[f"{prefix}_date"].dt.month.astype("Int64")
          elif check == "_day":
            date_value = df[f"{prefix}_date"].dt.day.astype("Int64")
          if not df[f"{prefix}{check}"].astype("Int64").equals(date_value):
            n_diff = df[f"{prefix}{check}"].astype("Int64").ne(date_value).sum()
            if n_diff > 0:
              raise ValueError(f"Derived field '{prefix}{check}' does not match the date field '{prefix}_date' in {n_diff} rows.")
        elif check in ["_year_month", "_year_quarter"]:
          date_value = None
          if check == "_year_month":
            date_value = df[f"{prefix}_date"].dt.to_period("M").astype("str")
          elif check == "_year_quarter":
            date_value = df[f"{prefix}_date"].dt.to_period("Q").astype("str")
          if not df[f"{prefix}{check}"].equals(date_value):
            n_diff = df[f"{prefix}{check}"].ne(date_value).sum()
            raise ValueError(f"Derived field '{prefix}{check}' does not match the date field '{prefix}_date' in {n_diff} rows.")
  return df


def _boolify_series(series: pd.Series):
  """
  Convert a series with potential string representations of booleans into actual booleans.

  :param series: Input series.
  :type series: pandas.Series
  :returns: Boolean series.
  :rtype: pandas.Series
  """
  if series.dtype in ["object", "string", "str"]:
    series = series.astype(str).str.lower().str.strip()
    series = series.replace(["true", "t", "1"], 1)
    series = series.replace(["false", "f", "0"], 0)
  series = series.fillna(0)
  series = series.astype(bool)
  return series


def _boolify_column_in_df(df: pd.DataFrame, field: str):
  """
  Convert a specified column in a DataFrame to boolean.

  :param df: Input DataFrame.
  :type df: pandas.DataFrame
  :param field: Column name to convert.
  :type field: str
  :returns: DataFrame with the specified column converted.
  :rtype: pandas.DataFrame
  """
  series = df[field]
  series = _boolify_series(series)
  df[field] = series
  return df


def _enrich_universe_spatial_lag(df_univ_in: pd.DataFrame, settings: dict, verbose: bool = False) -> pd.DataFrame:

  df = df_univ_in.copy()

  if "floor_area_ratio" not in df:
    df["floor_area_ratio"] = div_field_z_safe(df["bldg_area_finished_sqft"], df["land_area_sqft"])
  if "bedroom_density" not in df and "bldg_rooms_bed" in df:
    df["bedroom_density"] = div_field_z_safe(df["bldg_rooms_bed"], df["land_area_sqft"])

  value_fields = ["floor_area_ratio", "bedroom_density", "bldg_age_years", "dist_to_water_bodies", "dist_to_universities"]

  # Build a cKDTree from df_sales coordinates
  coords = df[['latitude', 'longitude']].values
  tree = cKDTree(coords)

  for value_field in value_fields:
    if value_field not in df:
      continue

    # Choose the number of nearest neighbors to use
    k = 5  # You can adjust this number as needed

    # Query the tree: for each parcel in df_universe, find the k nearest parcels
    # distances: shape (n_universe, k); indices: corresponding indices in df_sales
    distances, indices = tree.query(coords, k=k)

    # Ensure that distances and indices are 2D arrays (if k==1, reshape them)
    if k == 1:
      distances = distances[:, None]
      indices = indices[:, None]

    # For each universe parcel, compute sigma as the mean distance to its k neighbors.
    sigma = distances.mean(axis=1, keepdims=True)

    # Handle zeros in sigma
    sigma[sigma == 0] = np.finfo(float).eps  # Avoid division by zero

    # Compute Gaussian kernel weights for all neighbors
    weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))

    # Normalize the weights so that they sum to 1 for each parcel
    weights_norm = weights / weights.sum(axis=1, keepdims=True)

    # Get the values corresponding to the neighbor indices
    parcel_values = df[value_field].values
    neighbor_values = parcel_values[indices]  # shape (n_universe, k)

    # Compute the weighted average (spatial lag) for each parcel in the universe
    spatial_lag = (np.asarray(weights_norm) * np.asarray(neighbor_values)).sum(axis=1)

    # Add the spatial lag as a new column
    df[f"spatial_lag_{value_field}"] = spatial_lag

    median_value = df[value_field].median()
    df[f"spatial_lag_{value_field}"] = df[f"spatial_lag_{value_field}"].fillna(median_value)

  return df




def enrich_sup_spatial_lag(sup: SalesUniversePair, settings: dict, verbose: bool = False) -> SalesUniversePair:

  df_sales = sup.sales.copy()
  df_universe = sup.universe.copy()

  df_hydrated = get_hydrated_sales_from_sup(sup)

  sale_field = get_sale_field(settings)
  sale_field_vacant = f"{sale_field}_vacant"

  per_land_field = f"{sale_field}_land_sqft"
  per_impr_field = f"{sale_field}_impr_sqft"

  if per_land_field not in df_hydrated:
    df_hydrated[per_land_field] = div_field_z_safe(df_hydrated[sale_field], df_hydrated["land_area_sqft"])
    df_hydrated[per_land_field] = div_field_z_safe(df_hydrated[sale_field], df_hydrated["land_area_sqft"])
  if per_impr_field not in df_hydrated:
    df_hydrated[per_impr_field] = div_field_z_safe(df_hydrated[sale_field], df_hydrated["bldg_area_finished_sqft"])
    df_hydrated[per_impr_field] = div_field_z_safe(df_hydrated[sale_field], df_hydrated["bldg_area_finished_sqft"])
  if sale_field_vacant not in df_hydrated:
    df_hydrated[sale_field_vacant] = None
    df_hydrated[sale_field_vacant] = None
    df_hydrated[sale_field_vacant] = df_hydrated[sale_field].where(df_hydrated["bldg_area_finished_sqft"].le(0) & df_hydrated["land_area_sqft"].gt(0))
    df_hydrated[sale_field_vacant] = df_hydrated[sale_field].where(df_hydrated["bldg_area_finished_sqft"].le(0) & df_hydrated["land_area_sqft"].gt(0))

  value_fields = [sale_field, sale_field_vacant, per_land_field, per_impr_field]

  for value_field in value_fields:

    if value_field == sale_field:
      df_sub = df_hydrated.loc[df_hydrated["valid_sale"].eq(True)].copy()
    elif value_field == sale_field_vacant:
      df_sub = df_hydrated.loc[df_hydrated["valid_sale"].eq(True) & df_hydrated["bldg_area_finished_sqft"].le(0) & df_hydrated["land_area_sqft"].gt(0)].copy()
    elif value_field == per_land_field:
      df_sub = df_hydrated.loc[df_hydrated["valid_sale"].eq(True) & df_hydrated["bldg_area_finished_sqft"].le(0) & df_hydrated["land_area_sqft"].gt(0)].copy()
    elif value_field == per_impr_field:
      df_sub = df_hydrated.loc[df_hydrated["valid_sale"].eq(True) & df_hydrated["bldg_area_finished_sqft"].gt(0)].copy()
    else:
      raise ValueError(f"Unknown value field: {value_field}")

    if df_sub.empty:
      df_universe[f"spatial_lag_{value_field}"] = 0
      df_sales[f"spatial_lag_{value_field}"] = 0
      continue

    df_sub = df_sub[
      ~pd.isna(df_sub["latitude"]) &
      ~pd.isna(df_sub["longitude"])
    ]

    # Build a cKDTree from df_sales coordinates
    sales_coords = df_sub[['latitude', 'longitude']].values
    sales_tree = cKDTree(sales_coords)

    # Choose the number of nearest neighbors to use
    k = 5  # You can adjust this number as needed

    # Get the coordinates for the universe parcels
    universe_coords = df_universe[['latitude', 'longitude']].values

    # count any NA coordinates in the universe
    n_na_coords = universe_coords.shape[0] - np.count_nonzero(pd.isna(universe_coords).any(axis=1))
    print(f"Number of parcels in universe with coordinates: {n_na_coords} / {universe_coords.shape[0]}")

    # Query the tree: for each parcel in df_universe, find the k nearest sales
    # distances: shape (n_universe, k); indices: corresponding indices in df_sales
    distances, indices = sales_tree.query(universe_coords, k=k)

    # Ensure that distances and indices are 2D arrays (if k==1, reshape them)
    if k == 1:
      distances = distances[:, None]
      indices = indices[:, None]

    # For each universe parcel, compute sigma as the mean distance to its k neighbors.
    sigma = distances.mean(axis=1, keepdims=True)

    # Handle zeros in sigma
    sigma[sigma == 0] = np.finfo(float).eps  # Avoid division by zero

    # Compute Gaussian kernel weights for all neighbors
    weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))

    # Normalize the weights so that they sum to 1 for each parcel
    weights_norm = weights / weights.sum(axis=1, keepdims=True)

    # Get the sales prices corresponding to the neighbor indices
    sales_prices = df_sub[value_field].values
    neighbor_prices = sales_prices[indices]  # shape (n_universe, k)

    # Compute the weighted average (spatial lag) for each parcel in the universe
    spatial_lag = (np.asarray(weights_norm) * np.asarray(neighbor_prices)).sum(axis=1)

    # Add the spatial lag as a new column
    df_universe[f"spatial_lag_{value_field}"] = spatial_lag

    median_value = df_sub[value_field].median()
    df_universe[f"spatial_lag_{value_field}"] = df_universe[f"spatial_lag_{value_field}"].fillna(median_value)

    # Add the new field to sales:
    df_sales = df_sales.merge(df_universe[["key", f"spatial_lag_{value_field}"]], on="key", how="left")

  sup.set("sales", df_sales)
  sup.set("universe", df_universe)
  return sup


def _enrich_df_basic(df_in: pd.DataFrame, s_enrich_this: dict, dataframes: dict[str, pd.DataFrame], settings: dict, is_sales: bool = False, verbose: bool = False) -> pd.DataFrame:
  """
  Perform basic enrichment on a DataFrame including reference table joins, calculations,
  year built enrichment, and vacant status enrichment.

  :param df_in: Input DataFrame.
  :type df_in: pandas.DataFrame
  :param s_enrich_this: Enrichment instructions.
  :type s_enrich_this: dict
  :param dataframes: Dictionary of additional DataFrames.
  :type dataframes: dict[str, pd.DataFrame]
  :param settings: Settings dictionary.
  :type settings: dict
  :param is_sales: If True, indicates sales data.
  :type is_sales: bool, optional
  :param verbose: If True, prints progress.
  :type verbose: bool, optional
  :returns: Enriched DataFrame.
  :rtype: pandas.DataFrame
  """
  df = df_in.copy()
  s_ref = s_enrich_this.get("ref_tables", [])
  s_calc = s_enrich_this.get("calc", {})
  s_tweak = s_enrich_this.get("tweak", {})

  # reference tables:
  df = _perform_ref_tables(df, s_ref, dataframes, verbose=verbose)

  # calculations:
  df = perform_calculations(df, s_calc)

  # tweaks:
  df = perform_tweaks(df, s_tweak)

  # enrich year built:
  df = _enrich_year_built(df, settings, is_sales)

  return df


def _finesse_columns(df_in: pd.DataFrame | gpd.GeoDataFrame, suffix_left: str, suffix_right: str):
  """
  Combine columns with matching base names but different suffixes into a single column.

  :param df_in: Input DataFrame or GeoDataFrame.
  :type df_in: pandas.DataFrame or geopandas.GeoDataFrame
  :param suffix_left: Suffix of the left-hand columns.
  :type suffix_left: str
  :param suffix_right: Suffix of the right-hand columns.
  :type suffix_right: str
  :returns: DataFrame with combined columns.
  :rtype: pandas.DataFrame or geopandas.GeoDataFrame
  """
  df = df_in.copy()
  cols_to_finesse = []
  for col in df.columns.values:
    if col.endswith(suffix_left):
      base_col = col[:-len(suffix_left)]
      if base_col not in cols_to_finesse:
        cols_to_finesse.append(base_col)
  for col in cols_to_finesse:
    col_spatial = f"{col}{suffix_left}"
    col_data = f"{col}{suffix_right}"
    if col_spatial in df and col_data in df:
      df[col] = df[col_spatial].combine_first(df[col_data])
      df = df.drop(columns=[col_spatial, col_data], errors="ignore")
  return df


def _enrich_vacant(df_in: pd.DataFrame, settings:dict) -> pd.DataFrame:
  """
  Enrich the DataFrame by determining vacant properties based on finished building area.

  :param df_in: Input DataFrame.
  :type df_in: pandas.DataFrame
  :returns: DataFrame with an added 'is_vacant' column.
  :rtype: pandas.DataFrame
  """

  if "bldg_area_finished_sqft" in df_in:
    df = df_in.copy()
    df["is_vacant"] = False
    df.loc[pd.isna(df["bldg_area_finished_sqft"]), "bldg_area_finished_sqft"] = 0
    df.loc[df["bldg_area_finished_sqft"].eq(0), "is_vacant"] = True

    idx_vacant = df["is_vacant"].eq(True)

    # Remove building characteristics from anything that is vacant:
    df = simulate_removed_buildings(df, settings, idx_vacant)

  else:
    df = df_in

  return df


def _enrich_df_geometry(df_in: pd.DataFrame, s_enrich_this: dict, dataframes: dict[str, pd.DataFrame], settings: dict, verbose: bool = False) -> gpd.GeoDataFrame:
  """
  Perform basic geometric enrichment on a DataFrame by adding spatial features.

  :param df_in: Input DataFrame.
  :type df_in: pandas.DataFrame
  :param s_enrich_this: Enrichment instructions.
  :type s_enrich_this: dict
  :param dataframes: Dictionary of additional DataFrames.
  :type dataframes: dict[str, pd.DataFrame]
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints progress.
  :type verbose: bool, optional
  :returns: GeoDataFrame with enriched spatial features.
  :rtype: geopandas.GeoDataFrame
  """

  df = df_in.copy()
  s_geom = s_enrich_this.get("geometry", [])
  s_dist = s_enrich_this.get("distances", {})
  s_infer = s_enrich_this.get("infer", {})
  s_overture = s_enrich_this.get("overture", {})

  gdf_out = get_cached_df(df_in, "geom/enrich", "key", s_enrich_this)
  if gdf_out is not None:
    if verbose:
      print("--> found cached data...")
    return gdf_out

  gdf: gpd.GeoDataFrame

  # geometry
  gdf = _perform_spatial_joins(s_geom, dataframes, verbose=verbose)

  # Enrich with Overture building data if enabled
  if s_overture.get("enabled", False):

    if verbose:
      print("Enriching with Overture building data...")
    
    # Initialize Overture service with the correct settings path
    overture_settings = {
      "overture": s_overture  # Pass the overture settings directly
    }
    overture_service = init_service_overture(overture_settings)
    
    # Get bounding box from data
    bbox = gdf.to_crs("EPSG:4326").total_bounds
    
    # Fetch building data
    buildings = overture_service.get_buildings(bbox, use_cache=s_overture.get("cache", True), verbose=verbose)
    
    if not buildings.empty:
      # Calculate building footprints
      s_footprint = s_overture.get("footprint", {})
      footprint_units = s_footprint.get("units", None)
      if footprint_units is None:
        warnings.warn("`process.enrich.overture.footprint.units` not specified, defaulting to 'sqft'")
        footprint_units = "sqft"
      footprint_field = s_footprint.get("field", None)
      if footprint_field is None:
        warnings.warn("`process.enrich.overture.footprint.field` not specified, defaulting to 'bldg_area_footprint_sqft'")
        footprint_field = "bldg_area_footprint_sqft"
      gdf = overture_service.calculate_building_footprints(gdf, buildings, footprint_units, footprint_field, verbose=verbose)
    elif verbose:
      print("--> No buildings found in the area")

  # distances
  gdf = _perform_distance_calculations(gdf, s_dist, dataframes, get_long_distance_unit(settings), verbose=verbose)

  # Merge everything together:
  try_keys = ["key", "key2", "key3"]
  success = False
  gdf_merged: gpd.GeoDataFrame | None = None
  for key in try_keys:
    if key in gdf and key in df:
      if verbose:
        print(f"Using \"{key}\" to merge shapefiles onto df")
      n_dupes_gdf = gdf.duplicated(subset=key).sum()
      n_dupes_df = df.duplicated(subset=key).sum()
      if n_dupes_gdf > 0 or n_dupes_df > 0:
        raise ValueError(f"Found {n_dupes_gdf} duplicate keys in the geo_parcels dataframe, and {n_dupes_df} duplicate keys in the base dataframe. Cannot perform spatial join. De-duplicate your dataframes and try again.")
      gdf_merged = gdf.merge(df, on=key, how="left", suffixes=("_spatial", "_data"))
      gdf_merged = _finesse_columns(gdf_merged, "_spatial", "_data")
      success = True
      break
  if not success:
    raise ValueError(f"Could not find a common key between geo_parcels and base dataframe. Tried keys: {try_keys}")

  # spatially infer missing
  gdf_merged = perform_spatial_inference(gdf_merged, s_infer, "key", verbose=verbose)

  write_cached_df(df_in, gdf_merged, "geom/enrich", "key", s_enrich_this)

  return gdf_merged


def _enrich_polar_coordinates(gdf_in: gpd.GeoDataFrame, settings: dict, verbose: bool = False) -> gpd.GeoDataFrame:
  gdf = gdf_in[["key", "geometry"]].copy()

  longitude, latitude = get_center(settings, gdf)

  crs = get_crs(gdf, "equal_area")
  gdf = gdf.to_crs(crs)

  # convert longitude, latitude, to same point space as gdf:
  point = Point(longitude, latitude)
  single_point_gdf = gpd.GeoDataFrame({'geometry': [point]}, crs=gdf_in.crs)
  single_point_gdf = single_point_gdf.to_crs(crs)

  x_center = single_point_gdf.geometry.x.iloc[0]
  y_center = single_point_gdf.geometry.y.iloc[0]

  gdf["x_diff"] = gdf.geometry.centroid.x - x_center
  gdf["y_diff"] = gdf.geometry.centroid.y - y_center

  gdf['polar_radius'] = np.sqrt(gdf['x_diff']**2 + gdf['y_diff']**2)
  gdf['polar_angle'] = np.arctan2(gdf['y_diff'], gdf['x_diff'])
  gdf['polar_angle'] = np.degrees(gdf['polar_angle'])

  gdf_result = gdf_in.merge(gdf[["key", "polar_radius", "polar_angle"]], on="key", how="left")
  return gdf_result


def _basic_geo_enrichment(gdf_in: gpd.GeoDataFrame, settings: dict, verbose: bool = False) -> gpd.GeoDataFrame:
  """
  Perform basic geometric enrichment on a GeoDataFrame by adding spatial features.

  Adds latitude, longitude, GIS area, and calculates differences between given and GIS areas.
  Also counts vertices per parcel and computes additional geometric properties.

  :param gdf: Input GeoDataFrame.
  :type gdf: geopandas.GeoDataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints progress messages.
  :type verbose: bool, optional
  :returns: Enriched GeoDataFrame.
  :rtype: geopandas.GeoDataFrame
  """
  t = TimingData()

  if verbose:
    print(f"Performing basic geometric enrichment...")
  gdf_out = get_cached_df(gdf_in, "geom/basic", "key")
  if gdf_out is not None:
    if verbose:
      print("--> found cached data...")
    return gdf_out

  gdf = gdf_in.copy()

  t.start("latlon")
  gdf_latlon = gdf.to_crs(get_crs(gdf, "latlon"))
  gdf_area = gdf.to_crs(get_crs(gdf, "equal_area"))
  gdf["latitude"] = gdf_latlon.geometry.centroid.y
  gdf["longitude"] = gdf_latlon.geometry.centroid.x
  gdf["latitude_norm"] = (gdf["latitude"] - gdf["latitude"].min()) / (gdf["latitude"].max() - gdf["latitude"].min())
  gdf["longitude_norm"] = (gdf["longitude"] - gdf["longitude"].min()) / (gdf["longitude"].max() - gdf["longitude"].min())
  t.stop("latlon")
  if verbose:
    _t = t.get("latlon")
    print(f"--> added latitude/longitude...({_t:.2f}s)")
  t.start("area")
  gdf["land_area_gis_sqft"] = gdf_area.geometry.area
  gdf["land_area_given_sqft"] = gdf["land_area_sqft"]
  gdf["land_area_sqft"] = gdf["land_area_sqft"].combine_first(gdf["land_area_gis_sqft"])
  gdf["land_area_gis_delta_sqft"] = gdf["land_area_gis_sqft"] - gdf["land_area_sqft"]
  gdf["land_area_gis_delta_percent"] = div_field_z_safe(gdf["land_area_gis_delta_sqft"], gdf["land_area_sqft"])
  t.stop("area")
  if verbose:
    _t = t.get("area")
    print(f"--> calculated GIS area of each parcel...({_t:.2f}s)")
  gdf = _calc_geom_stuff(gdf, verbose)
  t.start("polar")
  gdf = _enrich_polar_coordinates(gdf, settings, verbose)
  t.stop("polar")
  if verbose:
    _t = t.get("polar")
    print(f"--> calculated polar coordinates...({_t:.2f}s)")

  write_cached_df(gdf_in, gdf, "geom/basic", "key")

  return gdf


def _calc_geom_stuff(gdf_in: gpd.GeoDataFrame, verbose: bool = False) -> gpd.GeoDataFrame:
  """
  Compute additional geometric properties for a GeoDataFrame, such as rectangularity and aspect ratio.

  :param gdf: Input GeoDataFrame.
  :type gdf: geopandas.GeoDataFrame
  :param verbose: If True, prints progress information.
  :type verbose: bool, optional
  :returns: GeoDataFrame with added properties.
  :rtype: geopandas.GeoDataFrame
  """

  print("HO")
  gdf = get_cached_df(gdf_in, "geom/stuff", "key")
  if gdf is not None:
    return gdf
  print("YO")

  t = TimingData()
  t.start("rectangularity")
  gdf = gdf_in.copy()
  min_rotated_rects = gdf.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
  min_rotated_rects_area_delta = np.abs(min_rotated_rects.area - gdf.geometry.area)
  min_rotated_rects_area_delta_percent = div_field_z_safe(min_rotated_rects_area_delta, gdf.geometry.area)
  gdf["geom_rectangularity_num"] = 1.0 - min_rotated_rects_area_delta_percent
  coords = min_rotated_rects.apply(lambda rect: np.array(rect.exterior.coords[:-1]))  # Drop duplicate last point
  t.stop("rectangularity")
  if verbose:
    _t = t.get("rectangularity")
    print(f"--> calculated parcel rectangularity...({_t:.2f}s)")
  t.start("aspect_ratio")
  edge_lengths = coords.apply(lambda pts: np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)))
  dimensions = edge_lengths.apply(lambda lengths: np.sort(lengths)[:2])
  aspect_ratios = dimensions.apply(lambda dims: dims[1] / dims[0] if dims[0] != 0 else float('inf'))
  gdf["geom_aspect_ratio"] = aspect_ratios
  t.stop("aspect_ratio")
  if verbose:
    _t = t.get("aspect_ratio")
    print(f"--> calculated parcel aspect ratios...({_t:.2f}s)")
  gdf = identify_irregular_parcels(gdf, verbose)

  write_cached_df(gdf_in, gdf, "geom/stuff", "key")
  return gdf


def _perform_spatial_joins(s_geom: list, dataframes: dict[str, pd.DataFrame], verbose: bool = False) -> gpd.GeoDataFrame:
  """
  Perform spatial joins based on a list of spatial join instructions.

  Strings in s_geom are interpreted as IDs of loaded shapefiles; dicts must contain an 'id'
  and optionally a 'predicate' (default "contains_centroid").

  :param s_geom: List of spatial join instructions.
  :type s_geom: list
  :param dataframes: Dictionary of DataFrames containing spatial data.
  :type dataframes: dict[str, pd.DataFrame]
  :param verbose: If True, prints progress messages.
  :type verbose: bool, optional
  :returns: GeoDataFrame after performing spatial joins.
  :rtype: geopandas.GeoDataFrame
  :raises ValueError: If required spatial data is missing.
  """
  if not isinstance(s_geom, list):
    s_geom = [s_geom]

  if "geo_parcels" not in dataframes:
    raise ValueError("No 'geo_parcels' dataframe found in the dataframes. This layer is required, and it must contain parcel geometry.")

  gdf_parcels: gpd.GeoDataFrame = dataframes["geo_parcels"]
  gdf_merged = gdf_parcels.copy()

  if verbose:
    print(f"Performing spatial joins...")

  for geom in s_geom:
    if isinstance(geom, str):
      entry = {"id": str(geom), "predicate": "contains_centroid"}
    elif isinstance(geom, dict):
      entry = geom
    else:
      raise ValueError(f"Invalid geometry entry: {geom}")
    _id = entry.get("id")
    predicate = entry.get("predicate", "contains_centroid")
    if _id is None:
      raise ValueError("No 'id' found in geometry entry.")
    if verbose:
      if predicate != "contains_centroid":
        print(f"--> {_id} @ {predicate}")
      else:
        print(f"--> {_id}")
    gdf = dataframes[_id]
    fields_to_tag = entry.get("fields", None)
    if fields_to_tag is None:
      fields_to_tag = [field for field in gdf.columns if field != "geometry"]
    else:
      for field in fields_to_tag:
        if field not in gdf:
          raise ValueError(f"Field to tag '{field}' not found in geometry dataframe '{_id}'.")
    gdf_merged = _perform_spatial_join(gdf_merged, gdf, predicate, fields_to_tag)

  gdf_no_geometry = gdf_merged[gdf_merged["geometry"].isna()]
  if len(gdf_no_geometry) > 0:
    warnings.warn(f"Found {len(gdf_no_geometry)} parcels with no geometry. These parcels will be excluded from the analysis. You can find them in out/errors/")
    os.makedirs("out/errors", exist_ok=True)
    gdf_no_geometry.to_parquet("out/errors/parcels_no_geometry.parquet")
    gdf_no_geometry.to_csv("out/errors/parcels_no_geometry.csv", index=False)
    gdf_no_geom_keys = gdf_no_geometry["key"].values
    with open("out/errors/parcels_no_geometry_keys.txt", "w") as f:
      for key in gdf_no_geom_keys:
        f.write(f"{key}\n")
    gdf_merged = gdf_merged.dropna(subset=["geometry"])
  return gdf_merged


def _perform_spatial_join_contains_centroid(gdf: gpd.GeoDataFrame, gdf_overlay: gpd.GeoDataFrame):
  """
  Perform a spatial join where the centroid of geometries in gdf is within gdf_overlay.

  :param gdf: Base GeoDataFrame.
  :type gdf: geopandas.GeoDataFrame
  :param gdf_overlay: Overlay GeoDataFrame.
  :type gdf_overlay: geopandas.GeoDataFrame
  :returns: GeoDataFrame after spatial join.
  :rtype: geopandas.GeoDataFrame
  """
  # Compute centroids of each parcel
  gdf["geometry_centroid"] = gdf.geometry.centroid

  # Use within predicate for spatial join
  gdf = gpd.sjoin(
    gdf.set_geometry("geometry_centroid"),
    gdf_overlay,
    how="left",
    predicate="within"
  )
  # remove extra columns like "index_right":
  gdf = gdf.drop(columns=["index_right"], errors="ignore")
  return gdf


def _perform_spatial_join(gdf_in: gpd.GeoDataFrame, gdf_overlay: gpd.GeoDataFrame, predicate: str, fields_to_tag: list[str]):
  """
  Perform a spatial join between two GeoDataFrames using the specified predicate.

  :param gdf_in: Base GeoDataFrame.
  :type gdf_in: geopandas.GeoDataFrame
  :param gdf_overlay: Overlay GeoDataFrame.
  :type gdf_overlay: geopandas.GeoDataFrame
  :param predicate: Spatial predicate to use (e.g., "contains_centroid").
  :type predicate: str
  :param fields_to_tag: List of fields to merge from the overlay.
  :type fields_to_tag: list[str]
  :returns: GeoDataFrame after performing the spatial join.
  :rtype: geopandas.GeoDataFrame
  :raises ValueError: If an invalid predicate is provided.
  """
  gdf = gdf_in.copy()
  gdf_overlay = gdf_overlay.to_crs(gdf.crs)
  if "__overlay_id__" in gdf_overlay:
    raise ValueError("The overlay GeoDataFrame already contains a '__overlay_id__' column. This column is used internally by the spatial join function, and must not be present in the overlay GeoDataFrame.")
  gdf_overlay["__overlay_id__"] = range(len(gdf_overlay))
  # TODO: add more predicates as needed
  if predicate == "contains_centroid":
    gdf = _perform_spatial_join_contains_centroid(gdf, gdf_overlay)
  else:
    raise ValueError(f"Invalid spatial join predicate: {predicate}")
  gdf = gdf.drop(columns=fields_to_tag, errors="ignore")
  gdf = gdf.merge(gdf_overlay[["__overlay_id__"] + fields_to_tag], on="__overlay_id__", how="left")
  gdf.set_geometry("geometry", inplace=True)
  gdf = gdf.drop(columns=["geometry_centroid", "__overlay_id__"], errors="ignore")
  return gdf

def _do_perform_spatial_inference(df: pd.DataFrame, s_infer: dict, field: str, key_field: str, verbose: bool = False) -> pd.DataFrame:
    """
    Perform spatial inference using specified model
    
    Args:
        df: DataFrame containing data
        s_infer: Dictionary with inference settings
        field: Field to infer
        key_field: Primary key field
        verbose: Whether to print detailed information
        
    Returns:
        DataFrame with inferred values
    """
    if verbose:
        print(f"\n=== Starting inference for field '{field}' ===")
    
    # Get model settings
    proxies = s_infer.get("proxies", [])
    locations = s_infer.get("locations", [])
    group_by = s_infer.get("group_by", [])
    
    if not proxies:
        raise ValueError(f"No proxy fields specified for inference of {field}")
    
    # Initialize model
    model = get_inference_model("ratio_proxy")
    
    # Split data into training and inference sets
    filters = s_infer.get("filters", [])
    if filters:
        inference_mask = select_filter(df, filters)
        training_mask = ~inference_mask & df[field].notna()
    else:
        inference_mask = df[field].isna()
        training_mask = ~inference_mask
        
    df_train = df[training_mask].copy()
    df_to_infer = df[inference_mask].copy()
    
    if verbose:
        print(f"\nData split:")
        print(f"--> {len(df_to_infer):,} rows need inference")
        print(f"--> {len(df_train):,} rows available for training")

    # First try direct fill from known sources
    fill_fields = s_infer.get("fill", [])
    if fill_fields:
        if verbose:
            print(f"\nFilling {field} with known values from: {fill_fields}")
        for fill_field in fill_fields:
            if fill_field not in df.columns:
                warnings.warn(f"Fill field '{fill_field}' not found in dataframe")
                continue
            mask = df[field].isna() & df[fill_field].notna() & df[fill_field].gt(0)
            df.loc[mask, field] = df.loc[mask, fill_field]
            if verbose:
                print(f"--> Filled {mask.sum():,} values from {fill_field}")
    
    # Fit model
    if verbose:
        print("\nFitting model...")
    model.fit(df_train, field, {"proxies": proxies, "locations": locations, "group_by": group_by})
    
    # Evaluate model
    if verbose:
        print("\nEvaluating model performance:")
        metrics = model.evaluate(df_train, field)
        for metric, value in metrics.items():
            print(f"--> {metric}: {value:.4f}")
    
    # Make predictions
    if verbose:
        print("\nMaking predictions...")
    predictions = model.predict(df_to_infer)
    
    # Update DataFrame
    df = df.copy()
    df.loc[inference_mask, field] = predictions
    df[f"inferred_{field}"] = False
    df.loc[inference_mask, f"inferred_{field}"] = True

    # Final statistics
    final_missing = df[field].isna().sum()
    if verbose:
        print(f"\nFinal results:")
        print(f"--> {len(predictions):,} values inferred")
        print(f"--> {final_missing:,} values remain empty ({final_missing/len(df)*100:.1f}% of total)")

    return df

def _do_perform_distance_calculations(df_in: gpd.GeoDataFrame, gdf_in: gpd.GeoDataFrame, _id: str, max_distance: float = None, unit: str = "km") -> pd.DataFrame:
    """
    Perform a divide-by-zero-safe nearest neighbor spatial join to calculate distances.

    :param df_in: Base GeoDataFrame.
    :type df_in: geopandas.GeoDataFrame
    :param gdf_in: Overlay GeoDataFrame.
    :type gdf_in: geopandas.GeoDataFrame
    :param _id: Identifier used for naming the distance column.
    :type _id: str
    :param max_distance: Maximum distance to consider (in specified unit)
    :type max_distance: float, optional
    :param unit: Unit for distance conversion (default "km").
    :type unit: str, optional
    :returns: DataFrame with added distance and within_distance columns.
    :rtype: pandas.DataFrame
    :raises ValueError: If an unsupported unit is specified.
    """
    unit_factors = {"m": 1, "km": 0.001, "mile": 0.000621371, "ft": 3.28084}
    if unit not in unit_factors:
        raise ValueError(f"Unsupported unit '{unit}'")
    crs = get_crs(df_in, "equal_distance")

    # check for duplicate keys:
    if df_in.duplicated(subset="key").sum() > 0:
      # caching won't work if there's duplicate keys, and there shouldn't be any duplicate keys here anyways
      raise ValueError(f"Duplicate keys found before distance calculation for '{_id}.' This should not happen.")

    # construct a unique cache signature
    signature = {
      "crs": crs.name,
      "_id": _id,
      "max_distance": max_distance,
      "unit": unit,
      "df_in_len": len(df_in),
      "gdf_in_len": len(gdf_in),
      "df_cols": sorted(df_in.columns.tolist()),
      "gdf_cols": sorted(gdf_in.columns.tolist()),
      "gdf_hash": hash(gdf_in.geometry.to_wkb().sum()),
    }
    # check if we already have this distance calculation
    if check_cache(f"osm/distance_{_id}", signature, "df"):
      df_net_change = read_cache(f"osm/distance_{_id}", "df")
      df_out = df_in.merge(df_net_change, on="key", how="left")
      return df_out

    df_projected = df_in.to_crs(crs).copy()
    gdf_projected = gdf_in.to_crs(crs).copy()
    
    # Initialize dictionary to store new columns
    new_columns = {
        f"within_{_id}": pd.Series(False, index=df_projected.index),
        f"dist_to_{_id}": pd.Series(np.nan, index=df_projected.index)
    }
    
    if max_distance is not None:
        # Create buffer around features we're measuring distance to
        gdf_buffer = gdf_projected.copy()
        gdf_buffer.geometry = gdf_buffer.geometry.buffer(max_distance / unit_factors[unit])
        
        # Find parcels that intersect with the buffer
        parcels_within = gpd.sjoin(
            df_projected, 
            gdf_buffer, 
            how="inner", 
            predicate="intersects"
        )
        
        # Clean up any index_right column from the spatial join
        parcels_within = parcels_within.drop(columns=["index_right"], errors="ignore")
        
        # Only calculate distances for parcels within buffer
        if len(parcels_within) > 0:
            nearest = gpd.sjoin_nearest(
                parcels_within,
                gdf_projected,
                how="left",
                distance_col=f"dist_to_{_id}"
            )
            
            # Clean up any index_right column from the spatial join
            nearest = nearest.drop(columns=["index_right"], errors="ignore")
            
            # Keep only the columns we need
            nearest = nearest[["key", f"dist_to_{_id}"]]
            
            nearest[f"dist_to_{_id}"] *= unit_factors[unit]
            
            # Mark these parcels as within distance
            new_columns[f"within_{_id}"] = pd.Series(False, index=df_projected.index)
            new_columns[f"within_{_id}"].loc[df_projected["key"].isin(parcels_within["key"])] = True
            
            # Handle duplicates in nearest
            if nearest.duplicated(subset="key").sum() > 0:
                nearest = nearest.sort_values(by=["key", f"dist_to_{_id}"], ascending=[True, True])
                nearest = nearest.drop_duplicates(subset="key")
            
            # Add distance column
            distances_series = pd.Series(nearest.set_index("key")[f"dist_to_{_id}"])
            new_columns[f"dist_to_{_id}"] = distances_series.reindex(df_projected["key"]).values
            
    else:
        # If no max_distance specified, calculate for all parcels
        nearest = gpd.sjoin_nearest(
            df_projected,
            gdf_projected,
            how="left",
            distance_col=f"dist_to_{_id}"
        )
        
        # Clean up any index_right column from the spatial join
        nearest = nearest.drop(columns=["index_right"], errors="ignore")
        
        # Keep only the columns we need
        nearest = nearest[["key", f"dist_to_{_id}"]]
        
        nearest[f"dist_to_{_id}"] *= unit_factors[unit]
        
        # Handle duplicates in nearest
        if nearest.duplicated(subset="key").sum() > 0:
            nearest = nearest.sort_values(by=["key", f"dist_to_{_id}"], ascending=[True, True])
            nearest = nearest.drop_duplicates(subset="key")
        
        # All parcels considered "within distance" when no max_distance specified
        new_columns[f"within_{_id}"] = pd.Series(True, index=df_projected.index)
        
        # Add distance column
        distances_series = pd.Series(nearest.set_index("key")[f"dist_to_{_id}"])
        new_columns[f"dist_to_{_id}"] = distances_series.reindex(df_projected["key"]).values
    
    # Create new DataFrame with all new columns
    new_df = pd.DataFrame(new_columns, index=df_projected.index)
    
    # Combine original DataFrame with new columns using concat
    df_out = pd.concat([df_in, new_df], axis=1)

    # Figure out what the net change was and cache that
    new_columns = [col for col in new_df.columns if col not in df_in.columns]
    if len(new_columns) > 0:
      df_net_change = df_out[["key"] + new_columns].copy()

      # check for duplicate keys:
      if df_net_change.duplicated(subset="key").sum() > 0:
        raise ValueError(f"Duplicate keys found after distance calculation for '{_id}.' This should not happen.")

      # save to cache:
      write_cache(f"osm/distance_{_id}", df_net_change, signature, "df")

    return df_out


def _perform_distance_calculations(df_in: gpd.GeoDataFrame, s_dist: dict, dataframes: dict[str, pd.DataFrame], unit: str = "km", verbose: bool = False) -> gpd.GeoDataFrame:
    """
    Perform distance calculations based on enrichment instructions.

    :param df_in: Base GeoDataFrame.
    :type df_in: geopandas.GeoDataFrame
    :param s_dist: Distance calculation instructions.
    :type s_dist: dict
    :param dataframes: Dictionary of additional DataFrames.
    :type dataframes: dict[str, pd.DataFrame]
    :param unit: Unit for distance conversion (default "km").
    :type unit: str, optional
    :param verbose: If True, prints progress information.
    :type verbose: bool, optional
    :returns: GeoDataFrame with calculated distance fields.
    :rtype: geopandas.GeoDataFrame
    :raises ValueError: If a distance entry is invalid.
    """
    df = df_in.copy()
    if verbose:
        print(f"Performing distance calculations...")
    
    # Collect all distance calculations to apply at once
    all_distance_dfs = []

    # check for duplicate keys:
    if df_in.duplicated(subset="key").sum() > 0:
      # caching won't work if there's duplicate keys, and there shouldn't be any duplicate keys here anyways
      raise ValueError(f"Duplicate keys found before distance calculation. This should not happen.")

    signature = {
      "unit": unit,
      "crs": df_in.crs.name,
      "df_in_len": len(df_in),
      "df_cols": sorted(df_in.columns.tolist()),
      "s_dist": s_dist,
    }
    # check if we already have this distance calculation
    if check_cache(f"osm/distance_all", signature, "df"):
      df_net_change = read_cache(f"osm/distance_all", "df")
      df_out = df_in.merge(df_net_change, on="key", how="left")
      return df_out
    
    for entry in s_dist:
        if isinstance(entry, str):
            entry = {"id": str(entry)}
        elif not isinstance(entry, dict):
            raise ValueError(f"Invalid distance entry: {entry}")
            
        _id = entry.get("id")
        max_distance = entry.get("max_distance")  # Get max_distance from settings
        entry_unit = entry.get("unit", unit)  # Allow overriding unit per feature
        
        if _id is None:
            raise ValueError("No 'id' found in distance entry.")
        if _id not in dataframes:
            if verbose:
                print(f"--> Skipping {_id} - not found in dataframes (likely disabled in settings)")
            continue
            
        gdf = dataframes[_id]
        field = entry.get("field", None)
        
        if verbose:
            print(f"--> {_id}")
            if max_distance is not None:
                print(f"    max_distance: {max_distance} {entry_unit}")
        if field is None:
            if verbose:
                print(f"--> {_id} field is None")
            # Calculate distances for this feature
            distance_df = _do_perform_distance_calculations(df, gdf, _id, max_distance, entry_unit)
            # Extract only the new columns
            new_cols = [col for col in distance_df.columns if col not in df.columns]
            all_distance_dfs.append(distance_df[new_cols])
            if verbose:
                print(f"--> {_id} done")
        else:
            if verbose:
                print(f"--> {_id} field is {field}")
            uniques = gdf[field].unique()
            for unique in uniques:
                if pd.isna(unique):
                    continue
                gdf_subset = gdf[gdf[field].eq(unique)]
                # Calculate distances for this subset
                distance_df = _do_perform_distance_calculations(df, gdf_subset, f"{_id}_{unique}", max_distance, entry_unit)
                # Extract only the new columns
                new_cols = [col for col in distance_df.columns if col not in df.columns]
                all_distance_dfs.append(distance_df[new_cols])
            if verbose:
                print(f"--> {_id} done")
    
    # Apply all distance calculations at once
    if all_distance_dfs:
        # Combine all distance DataFrames
        combined_distances = pd.concat(all_distance_dfs, axis=1)
        # Combine with original DataFrame
        df = pd.concat([df, combined_distances], axis=1)

    new_cols = [col for col in df.columns if col not in df_in.columns]
    df_net_change = df[["key"]+new_cols].copy()
    # check for duplicate keys:
    if df_net_change.duplicated(subset="key").sum() > 0:
        raise ValueError(f"Duplicate keys found after distance calculation. This should not happen.")
    # save to cache:
    write_cache(f"osm/distance_all", df_net_change, signature, "df")

    return df


def _perform_ref_tables(df_in: pd.DataFrame | gpd.GeoDataFrame, s_ref: list | dict, dataframes: dict[str, pd.DataFrame], verbose: bool = False) -> pd.DataFrame | gpd.GeoDataFrame:
  """
  Perform reference table joins to enrich the input DataFrame.

  :param df_in: Input DataFrame or GeoDataFrame.
  :type df_in: pandas.DataFrame or geopandas.GeoDataFrame
  :param s_ref: Reference table instructions (list or dict).
  :type s_ref: list or dict
  :param dataframes: Dictionary of reference DataFrames.
  :type dataframes: dict[str, pd.DataFrame]
  :param verbose: If True, prints progress information.
  :type verbose: bool, optional
  :returns: Enriched DataFrame after reference table joins.
  :rtype: pandas.DataFrame or geopandas.GeoDataFrame
  :raises ValueError: If required keys or fields are missing.
  """
  df = df_in.copy()
  if not isinstance(s_ref, list):
    s_ref = [s_ref]

  if verbose:
    print(f"Performing reference table joins...")

  for ref in s_ref:
    _id = ref.get("id", None)
    key_ref_table = ref.get("key_ref_table", None)
    key_target = ref.get("key_target", None)
    add_fields = ref.get("add_fields", None)
    if verbose:
      print(f"--> {_id}")
    if _id is None:
      raise ValueError("No 'id' found in ref table.")
    if key_ref_table is None:
      raise ValueError("No 'key_ref_table' found in ref table.")
    if key_target is None:
      raise ValueError("No 'key_target' found in ref table.")
    if add_fields is None:
      raise ValueError("No 'add_fields' found in ref table.")
    if not isinstance(add_fields, list):
      raise ValueError("The 'add_fields' field must be a list of strings.")
    if len(add_fields) == 0:
      raise ValueError("The 'add_fields' field must contain at least one string.")
    if _id not in dataframes:
      raise ValueError(f"Ref table '{_id}' not found in dataframes.")
    df_ref = dataframes[_id]
    if key_ref_table not in df_ref:
      raise ValueError(f"Key field '{key_ref_table}' not found in ref table '{_id}'.")
    if key_target not in df:
      print(f"Target field '{key_target}' not found in base dataframe")
      print(f"base df columns = {df.columns.values}")
      raise ValueError(f"Target field '{key_target}' not found in base dataframe")
    for field in add_fields:
      if field not in df_ref:
        raise ValueError(f"Field '{field}' not found in ref table '{_id}'.")
      if field in df_in:
        raise ValueError(f"Field '{field}' already exists in base dataframe.")
    df_ref = df_ref[[key_ref_table] + add_fields]
    if key_ref_table == key_target:
      df = df.merge(df_ref, on=key_target, how="left")
    else:
      df = df.merge(df_ref, left_on=key_target, right_on=key_ref_table, how="left")
      df = df.drop(columns=[key_ref_table])
  return df


def _get_calc_cols(settings: dict, exclude_loaded_fields: bool = False) -> list[str]:
  """
  Retrieve a list of calculated columns based on settings.

  :param settings: Settings dictionary.
  :type settings: dict
  :returns: List of column names used in calculations.
  :rtype: list[str]
  """
  s_load = settings.get("data", {}).get("load", {})
  cols_found = []
  cols_base = []
  for key in s_load:
    entry = s_load[key]
    cols = _do_get_calc_cols(entry)
    cols_found += cols
    if exclude_loaded_fields:
      entry_load = entry.get("load", {})
      for load_key in entry_load:
        cols_base.append(load_key)

  cols_found = list(set(cols_found)-set(cols_base))
  return cols_found


def _do_get_calc_cols(df_entry: dict) -> list[str]:
  """
  Extract column names referenced in a calculation dictionary.

  :param df_entry: DataFrame entry from settings.
  :type df_entry: dict
  :returns: List of column names referenced in calculations.
  :rtype: list[str]
  """
  e_calc = df_entry.get("calc", {})
  fields_in_calc = _crawl_calc_dict_for_fields(e_calc)
  return fields_in_calc


def _load_dataframe(entry: dict, settings: dict, verbose: bool = False, fields_cat: list = None, fields_bool: list = None, fields_num: list = None) -> pd.DataFrame | None:
  """
  Load a DataFrame from a file based on instructions and perform calculations and type adjustments.

  :param entry: Dictionary with file loading instructions.
  :type entry: dict
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints progress information.
  :type verbose: bool, optional
  :param fields_cat: List of categorical fields.
  :type fields_cat: list, optional
  :param fields_bool: List of boolean fields.
  :type fields_bool: list, optional
  :param fields_num: List of numeric fields.
  :type fields_num: list, optional
  :returns: Loaded and processed DataFrame, or None if filename is empty.
  :rtype: pandas.DataFrame or None
  :raises ValueError: If an unsupported file extension is encountered.
  """
  filename = entry.get("filename", "")
  if filename == "":
    return None
  filename = f"in/{filename}"
  ext = str(filename).split(".")[-1]

  column_names = _snoop_column_names(filename)

  e_load = entry.get("load", {})

  # Get all calc and tweak operations in order they appear
  operation_order = []
  for key in entry:
    if "calc" in key or "tweak" in key:  # Match any key containing calc or tweak
      op_type = "calc" if "calc" in key else "tweak"
      operation_order.append({"type": op_type, "operations": entry[key]})

  if verbose:
    print(f"Loading \"{filename}\"...")

  rename_map = {}
  dtype_map = {}
  extra_map = {}
  cols_to_load = []

  for rename_key in e_load:
    original = e_load[rename_key]
    original_key = None
    if isinstance(original, list):
      if len(original) > 0:
        original_key = original[0]
        cols_to_load += [original_key]
        rename_map[original_key] = rename_key
      if len(original) > 1:
        dtype_map[original_key] = original[1]
        if original[1] == "datetime":
          dtype_map[original_key] = "str"
      if len(original) > 2:
        extra_map[rename_key] = original[2]
    elif isinstance(original, str):
      cols_to_load += [original]
      rename_map[original] = rename_key

  # Only include fields from calcs that exist in the source data
  fields_in_calc = []
  for operation in operation_order:
    if operation["type"] == "calc":
      fields_in_calc.extend(_crawl_calc_dict_for_fields(operation["operations"]))
  fields_in_calc = [f for f in fields_in_calc if f in column_names]
  cols_to_load += fields_in_calc
  cols_to_load = list(set(cols_to_load))

  is_geometry = False
  if "geometry" in column_names and "geometry" not in cols_to_load:
    cols_to_load.append("geometry")
    is_geometry = True

  if ext == "parquet":
    try:
      df = gpd.read_parquet(filename, columns=cols_to_load)
    except ValueError:
      df = pd.read_parquet(filename, columns=cols_to_load)

    # Enforce user's dtypes
    for col in df.columns:
      if col in dtype_map:
        df[col] = df[col].astype(dtype_map[col])

  elif ext == "csv":
    df = pd.read_csv(filename, usecols=cols_to_load, dtype=dtype_map)
  else:
    raise ValueError(f"Unsupported file extension: {ext}")

  # Rename columns
  df = df.rename(columns=rename_map)

  # Perform operations in order they appear in settings
  for operation in operation_order:
    op_type = operation["type"]
    if op_type == "calc":
      df = perform_calculations(df, operation["operations"], rename_map)
    elif op_type == "tweak":
      df = perform_tweaks(df, operation["operations"], rename_map)

  if fields_cat is None:
    fields_cat = get_fields_categorical(settings, include_boolean=False)
  if fields_bool is None:
    fields_bool = get_fields_boolean(settings)
  if fields_num is None:
    fields_num = get_fields_numeric(settings, include_boolean=False)

  for col in df.columns:
    if col in fields_cat:
      df[col] = df[col].astype("string")
    elif col in fields_bool:
      df[col] = _boolify_series(df[col])
    elif col in fields_num:
      df[col] = df[col].astype("Float64")

  date_fields = get_fields_date(settings, df)
  time_format_map = {}
  for xkey in extra_map:
    if xkey in date_fields:
      time_format_map[xkey] = extra_map[xkey]
  for dkey in date_fields:
    if dkey not in time_format_map:
      example_value = df[~df[dkey].isna()][dkey].iloc[0]
      raise ValueError(f"Date field '{dkey}' does not have a time format specified. Example value from {dkey}: \"{example_value}\"")
  df = enrich_time(df, time_format_map, settings)

  dupes = entry.get("dupes", None)
  dupes_was_none = dupes is None
  if dupes is None:
    if is_geometry:
      dupes = "auto"
    else:
      dupes = {}
  if dupes == "auto":
    if is_geometry:
      cols = [col for col in df.columns.values if col != "geometry"]
      col = cols[0]
      dupes = {"subset": [col], "sort_by": [col, "asc"], "drop": True}
      if dupes_was_none:
        warnings.warn(f"'dupes' not found for geo df '{filename}', defaulting to \"{col}\" as de-dedupe key. Set 'dupes:\"auto\" to remove this warning.'")
    else:
      keys = ["key", "key2", "key3"]
      for key in keys:
        if key in df:
          dupes = {"subset": [key], "sort_by": [key, "asc"], "drop": True}
          break

  df = _handle_duplicated_rows(df, dupes)

  if is_geometry:
    gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(df, geometry="geometry")
    gdf = clean_geometry(gdf, ensure_polygon=True)
    df = gdf

  drop = entry.get("drop", [])
  if len(drop) > 0:
    df = df.drop(columns=drop, errors="ignore")

  return df


def _snoop_column_names(filename: str) -> list[str]:
  """
  Retrieve column names from a file without loading full data.

  :param filename: Path to the file.
  :type filename: str
  :returns: List of column names.
  :rtype: list[str]
  :raises ValueError: If file extension is unsupported.
  """
  ext = str(filename).split(".")[-1]
  if ext == "parquet":
    parquet_file = pq.ParquetFile(filename)
    return parquet_file.schema.names
  elif ext == "csv":
    return pd.read_csv(filename, nrows=0).columns.tolist()
  raise ValueError(f"Unsupported file extension: \"{ext}\"")


def _handle_duplicated_rows(df_in: pd.DataFrame, dupes: str|dict, verbose: bool = False) -> pd.DataFrame:
  """
  Handle duplicated rows in a DataFrame based on specified rules.

  :param df_in: Input DataFrame.
  :type df_in: pandas.DataFrame
  :param dupes: Dictionary specifying duplicate handling instructions.
  :type dupes: dict
  :param verbose: If True, prints information.
  :type verbose: bool, optional
  :returns: DataFrame with duplicates handled.
  :rtype: pandas.DataFrame
  """
  if dupes == "allow":
    return df_in
  subset = dupes.get("subset", "key")
  if not isinstance(subset, list):
    subset = [subset]
  for key in subset:
    if key not in df_in:
      return df_in
  do_drop = dupes.get("drop", True)
  agg : dict | None = dupes.get("agg", None)
  num_dupes = df_in.duplicated(subset=subset).sum()
  orig_len = len(df_in)
  if num_dupes > 0:
    if agg is not None:
      df_agg : pd.DataFrame | None = None
      for agg_entry in agg:
        field = agg_entry.get("field")
        op = agg_entry.get("op")
        alias = agg_entry.get("alias", f"{field}_{op}")
        if field not in df_in:
          raise ValueError(f"Field '{field}' not found in DataFrame.")
        df_result = df_in.groupby(subset).agg({field: op}).reset_index().rename(columns={field: alias})
        if df_agg is None:
          df_agg = df_result
        else:
          df_agg = df_agg.merge(df_result, on=subset, how="outer")
      return df_agg
    else:
      sort_by = dupes.get("sort_by", ["key", "asc"])
      if not isinstance(sort_by, list):
        raise ValueError("sort_by must be a list of string pairs of the form [<field_name>, <asc|desc>]")
      if len(sort_by) == 2:
        if isinstance(sort_by[0], str) and isinstance(sort_by[1], str):
          sort_by = [sort_by]
      else:
        for entry in sort_by:
          if not isinstance(entry, list):
            raise ValueError(f"sort_by must be a list of string pairs, but found a non-list entry: {entry}")
          elif len(entry) != 2:
            raise ValueError(f"sort_by entry has {len(entry)} members: {entry}")
          elif not isinstance(entry[0], str) or not isinstance(entry[1], str):
            raise ValueError(f"sort_by entry has non-string members: {entry}")
      df = df_in.copy()
      bys = [x[0] for x in sort_by]
      ascendings = [x[1] == "asc" for x in sort_by]
      df = df.sort_values(by=bys, ascending=ascendings)
      if do_drop:
        if do_drop == "all":
          df = df.drop_duplicates(subset=subset, keep=False)
        else:
          df = df.drop_duplicates(subset=subset, keep="first")
        final_len = len(df)
        if verbose:
          print(f"Dropped {orig_len - final_len} duplicate rows based on '{subset}'")
      return df.reset_index(drop=True)
  return df_in


def _merge_dict_of_dfs(dataframes: dict[str, pd.DataFrame], merge_list: list, settings: dict, required_key="key") -> pd.DataFrame:
  """
  Merge multiple DataFrames according to merge instructions.

  :param dataframes: Dictionary mapping keys to DataFrames.
  :type dataframes: dict[str, pd.DataFrame]
  :param merge_list: List of merge instructions.
  :type merge_list: list
  :param settings: Settings dictionary.
  :type settings: dict
  :returns: Merged DataFrame.
  :rtype: pandas.DataFrame
  :raises ValueError: If required keys are missing.
  """
  merges = []
  s_reconcile = settings.get("data", {}).get("process", {}).get("reconcile", {})

  # Generate instructions for merging, but don't merge just yet
  for entry in merge_list:
    df_id = None
    how = "left"
    on = required_key
    left_on = None
    right_on = None

    payload = {}

    if isinstance(entry, str):
      if entry not in dataframes:
        raise ValueError(f"Merge key '{entry}' not found in dataframes.")
      df_id = entry
    elif isinstance(entry, dict):
      df_id = entry.get("id", None)
      how = entry.get("how", how)
      on = entry.get("on", on)
      left_on = entry.get("left_on", left_on)
      right_on = entry.get("right_on", right_on)
      for key in entry:
        if key not in ["id", "df", "how", "on", "left_on", "right_on"]:
          payload[key] = entry[key]
    if df_id is None:
      raise ValueError("Merge entry must be either a string or a dictionary with an 'id' key.")
    if df_id not in dataframes:
      raise ValueError(f"Merge key '{df_id}' not found in dataframes.")

    payload["id"] = df_id
    payload["df"] = dataframes[df_id]
    payload["how"] = how
    payload["on"] = on
    payload["left_on"] = left_on
    payload["right_on"] = right_on

    merges.append(payload)

  df_merged: pd.DataFrame | None = None
  all_cols = []
  conflicts = {}
  all_suffixes = []

  # Generate suffixes and note conflicts, which we'll resolve further down
  for merge in merges:
    df_id = merge["id"]
    df = merge["df"]
    on = merge["on"]
    how = merge["how"]
    left_on = merge["left_on"]
    right_on = merge["right_on"]
    merge_keys = []
    if on is not None:
      merge_keys = [on] if not isinstance(on, list) else on
    if right_on is not None:
      merge_keys = right_on if isinstance(right_on, list) else [right_on]
    if how == "lat_long":
      merge_keys = ["latitude", "longitude"]

    suffixes = {}
    for col in df.columns.values:
      if col in merge_keys:
        continue
      if col not in all_cols:
        all_cols.append(col)
      else:
        suffixed = f"{col}_{merge['id']}"
        suffixes[col] = suffixed
        if col not in conflicts:
          conflicts[col] = []
        conflicts[col].append(suffixed)
        all_suffixes.append(suffixed)
    df = df.rename(columns=suffixes)
    merge["df"] = df

  # Perform the actual merges
  for merge in merges:
    _id = merge["id"]
    df = merge.get("df", None)
    how = merge.get("how", "left")
    on = merge.get("on", required_key)
    left_on = merge.get("left_on", None)
    right_on = merge.get("right_on", None)
    dupes = merge.get("dupes", None)

    if df_merged is None:
      df_merged = df
    elif how == "append":
      df_merged = pd.concat([df_merged, df], ignore_index=True)
    elif how == "lat_long":
      if not (isinstance(df_merged, gpd.GeoDataFrame) and "geometry" in df_merged):
        raise ValueError("Cannot perform lat_long merge against a non-geodataframe. Make sure there is a geodataframe earlier in the merge chain.")
      if "latitude" not in df.columns and "longitude" not in df.columns:
        raise ValueError("Neither 'latitude' nor 'longitude' fields found in dataframe being merged with 'lat_long'")
      if "latitude" not in df.columns:
        raise ValueError("No 'latitude' field found in dataframe being merged with 'lat_long'")
      if "longitude" not in df.columns:
        raise ValueError("No 'longitude' field found in dataframe being merged with 'lat_long'")
      # use geolocation to get the right keys
      parcel_id_field = on if on is not None else "key"
      df_with_key = geolocate_point_to_polygon(df_merged, df, lat_field="latitude", lon_field="longitude", parcel_id_field=parcel_id_field)

      # de-duplicate
      dupe_rows = df_with_key[df_with_key.duplicated(subset=[parcel_id_field], keep=False)]
      if len(dupe_rows) > 0:
        if dupes is None:
          raise ValueError(f"Found {len(dupe_rows)} duplicates in geolocation merge '{_id}' on field '{parcel_id_field}'. But, you have no 'dupes' policy to deal with them. If you're okay with duplicates (such as in a sales dataset), set dupes='allow' in the merge instructions.")
        df_with_key = _handle_duplicated_rows(df_with_key, dupes, verbose=True)

      # merge the dataframes the conventional way
      df_merged = pd.merge(df_merged, df_with_key, how="left", on=parcel_id_field, suffixes=("", f"_{_id}"))
    else:
      if left_on is not None and right_on is not None:
        # Verify that both columns exist before attempting merge
        if isinstance(left_on, list):
          for col in left_on:
            if col not in df_merged.columns:
              raise ValueError(f"Left merge column '{col}' not found in left dataframe. Available columns: {df_merged.columns.tolist()}")
        else:
          if left_on not in df_merged.columns:
            raise ValueError(f"Left merge column '{left_on}' not found in left dataframe. Available columns: {df_merged.columns.tolist()}")
            
        if isinstance(right_on, list):
          for col in right_on:
            if col not in df.columns:
              raise ValueError(f"Right merge column '{col}' not found in right dataframe. Available columns: {df.columns.tolist()}")
        else:
          if right_on not in df.columns:
            raise ValueError(f"Right merge column '{right_on}' not found in right dataframe. Available columns: {df.columns.tolist()}")
            
        df_merged = pd.merge(df_merged, df, how=how, left_on=left_on, right_on=right_on, suffixes=("", f"_{_id}"))
      else:
        if on not in df_merged.columns:
          raise ValueError(f"Merge column '{on}' not found in left dataframe. Available columns: {df_merged.columns.tolist()}")
        if on not in df.columns:
          raise ValueError(f"Merge column '{on}' not found in right dataframe. Available columns: {df.columns.tolist()}")
        df_merged = pd.merge(df_merged, df, how=how, on=on, suffixes=("", f"_{_id}"))

    # General case de-duplication
    if on in df_merged:
      dupe_rows = df_merged[df_merged.duplicated(subset=[on], keep=False)]
      if len(dupe_rows) > 0:
        if dupes is None:
          raise ValueError(f"Found {len(dupe_rows)} duplicates in geolocation merge id='{_id}' how='{how}' on='{on}'. But, you have no 'dupes' policy to deal with them. If you're okay with duplicates (such as in a sales dataset), set dupes='allow' in the merge instructions.")
        df_merged = _handle_duplicated_rows(df_merged, dupes, verbose=True)

  # Reconcile conflicts
  for base_field in s_reconcile:
    df_ids = s_reconcile[base_field]
    if base_field not in all_cols:
      raise ValueError(f"Reconciliation field '{base_field}' not found in any of the dataframes.")
    child_fields = [f"{base_field}_{df_id}" for df_id in df_ids]
    if base_field in conflicts:
      old_child_fields = conflicts[base_field]
      old_child_fields = [field for field in old_child_fields if field not in child_fields]
      child_fields = child_fields + old_child_fields
    conflicts[base_field] = child_fields
  for base_field in conflicts:
    if base_field not in df_merged:
      warnings.warn(f"Reconciliation field '{base_field}' not found in merged dataframe.")
      continue
    child_fields = conflicts[base_field]
    if len(child_fields) > 1:
      #TODO: remove this when this becomes default pandas behavior
      old_value = pd.get_option('future.no_silent_downcasting')
      pd.set_option('future.no_silent_downcasting', True)

      df_merged[base_field] = df_merged[base_field].fillna(df_merged[child_fields[0]])
      for i in range(1, len(child_fields)):
        df_merged[base_field] = df_merged[base_field].fillna(df_merged[child_fields[i]])
      df_merged = df_merged.drop(columns=child_fields)

      #TODO: remove this when this becomes default pandas behavior
      pd.set_option('future.no_silent_downcasting', old_value)

  # Remove columns used as INGREDIENTS in calculations, but which the user never intends to load directly
  calc_cols = _get_calc_cols(settings, exclude_loaded_fields=True)
  for col in df_merged.columns.values:
    if col in calc_cols:
      df_merged = df_merged.drop(columns=[col])

  # Final checks
  if required_key is not None and required_key not in df_merged:
    raise ValueError(f"No '{required_key}' field found in merged dataframe. This field is required.")
  len_old = len(df_merged)
  df_merged = df_merged.dropna(subset=[required_key])
  len_new = len(df_merged)
  if len_new < len_old:
    warnings.warn(f"Dropped {len_old - len_new} rows due to missing primary key.")

  all_suffixes = [col for col in all_suffixes if col in df_merged]
  df_merged = df_merged.drop(columns=all_suffixes)

  # ensure a clean index:
  df_merged = df_merged.reset_index(drop=True)

  return df_merged


def _write_canonical_splits(sup: SalesUniversePair, settings: dict):
  """
  Write canonical split keys for sales data to disk.

  :param sup: SalesUniversePair containing sales and universe DataFrames.
  :type sup: SalesUniversePair
  :param settings: Settings dictionary.
  :type settings: dict
  :returns: None
  """
  df_sales_in = sup.sales
  df_univ = sup.universe
  df_sales = get_sales(df_sales_in, settings, df_univ=df_univ)
  model_groups = get_model_group_ids(settings, df_sales)
  instructions = settings.get("modeling", {}).get("instructions", {})
  test_train_frac = instructions.get("test_train_frac", 0.8)
  random_seed = instructions.get("random_seed", 1337)
  for model_group in model_groups:
    _do_write_canonical_split(model_group, df_sales, settings, test_train_frac, random_seed)


def _perform_canonical_split(model_group: str, df_sales_in: pd.DataFrame, settings: dict, test_train_fraction: float = 0.8, random_seed: int = 1337):
  """
  Perform a canonical split of the sales DataFrame for a given model group into test and training sets.

  :param model_group: Model group identifier.
  :type model_group: str
  :param df_sales_in: Input sales DataFrame.
  :type df_sales_in: pandas.DataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :param test_train_fraction: Fraction of data to use for training (default is 0.8).
  :type test_train_fraction: float, optional
  :param random_seed: Random seed for reproducibility (default is 1337).
  :type random_seed: int, optional
  :returns: Tuple of (test DataFrame, training DataFrame).
  :rtype: tuple(pandas.DataFrame, pandas.DataFrame)
  """
  df = df_sales_in[df_sales_in["model_group"].eq(model_group)].copy()
  df_v = get_vacant_sales(df, settings)
  df_i = df.drop(df_v.index)
  np.random.seed(random_seed)
  df_v_train = df_v.sample(frac=test_train_fraction)
  df_v_test = df_v.drop(df_v_train.index)
  df_i_train = df_i.sample(frac=test_train_fraction)
  df_i_test = df_i.drop(df_i_train.index)
  df_test = pd.concat([df_v_test, df_i_test]).reset_index(drop=True)
  df_train = pd.concat([df_v_train, df_i_train]).reset_index(drop=True)
  return df_test, df_train


def _do_write_canonical_split(model_group: str, df_sales_in: pd.DataFrame, settings: dict, test_train_fraction: float = 0.8, random_seed: int = 1337):
  """
  Write the canonical split keys (train and test) for a given model group to disk.

  :param model_group: Model group identifier.
  :type model_group: str
  :param df_sales_in: Input sales DataFrame.
  :type df_sales_in: pandas.DataFrame
  :param settings: Settings dictionary.
  :type settings: dict
  :param test_train_fraction: Fraction of data for training (default is 0.8).
  :type test_train_fraction: float, optional
  :param random_seed: Random seed for reproducibility (default is 1337).
  :type random_seed: int, optional
  :returns: None
  """
  df_test, df_train = _perform_canonical_split(model_group, df_sales_in, settings, test_train_fraction, random_seed)
  outpath = f"out/models/{model_group}/_data"
  os.makedirs(outpath, exist_ok=True)
  df_train[["key_sale"]].to_csv(f"{outpath}/train_keys.csv", index=False)
  df_test[["key_sale"]].to_csv(f"{outpath}/test_keys.csv", index=False)


def _read_split_keys(model_group: str):
  """
  Read the train and test split keys for a model group from disk.

  :param model_group: Model group identifier.
  :type model_group: str
  :returns: Tuple of (test keys, train keys) as numpy arrays.
  :rtype: tuple(numpy.ndarray, numpy.ndarray)
  :raises ValueError: If split key files are not found.
  """
  path = f"out/models/{model_group}/_data"
  train_path = f"{path}/train_keys.csv"
  test_path = f"{path}/test_keys.csv"
  if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise ValueError("No split keys found.")
  train_keys = pd.read_csv(train_path)["key_sale"].astype(str).values
  test_keys = pd.read_csv(test_path)["key_sale"].astype(str).values
  return test_keys, train_keys


def _tag_model_groups_sup(sup: SalesUniversePair, settings: dict, verbose: bool = False):
  """
  Tag model groups for both sales and universe DataFrames based on settings.

  Hydrates sales data and assigns model groups to parcels and sales by applying filters from settings.
  Also prints summary statistics if verbose is True.

  :param sup: SalesUniversePair containing sales and universe DataFrames.
  :type sup: SalesUniversePair
  :param settings: Settings dictionary.
  :type settings: dict
  :param verbose: If True, prints detailed progress information.
  :type verbose: bool, optional
  :returns: Updated SalesUniversePair with model group tags.
  :rtype: SalesUniversePair
  """
  df_sales = sup["sales"].copy()
  df_univ = sup["universe"].copy()
  df_sales_hydrated = get_hydrated_sales_from_sup(sup)
  mg = settings.get("modeling", {}).get("model_groups", {})

  print(f"Len univ before = {len(df_univ)}")
  print(f"Len sales before = {len(df_sales)} after = {len(df_sales_hydrated)}")
  print(f"Overall")
  print(f"--> {len(df_univ):,} parcels")
  print(f"--> {len(df_sales):,} sales")

  df_univ["model_group"] = None
  df_sales_hydrated["model_group"] = None

  for mg_id in mg:
    # only apply model groups to parcels that don't already have one
    idx_no_model_group = df_univ["model_group"].isnull()
    entry = mg[mg_id]
    _filter = entry.get("filter", [])

    if len(_filter) == 0:
      raise ValueError("No 'filter' entry found for model group '{mg_id}'. Check your spelling!")

    univ_index = resolve_filter(df_univ, _filter)
    df_univ.loc[idx_no_model_group & univ_index, "model_group"] = mg_id

    idx_no_model_group = df_sales_hydrated["model_group"].isnull()
    sales_index = resolve_filter(df_sales_hydrated, _filter)
    df_sales_hydrated.loc[idx_no_model_group & sales_index, "model_group"] = mg_id

  os.makedirs("out/look", exist_ok=True)

  if not isinstance(df_univ, gpd.GeoDataFrame):
    df_univ = gpd.GeoDataFrame(df_univ, geometry="geometry")
  df_univ.to_parquet("out/look/tag-univ-0.parquet")

  if not isinstance(df_univ, gpd.GeoDataFrame):
    df_univ = gpd.GeoDataFrame(df_univ, geometry="geometry")
  df_univ.to_parquet("out/look/tag-univ-0.parquet", engine="pyarrow")
  old_model_group = df_univ[["key", "model_group"]]

  for mg_id in mg:
    entry = mg[mg_id]
    print(f"Assigning model group {mg_id}...")
    common_area = entry.get("common_area", False)
    print("common_area --> ", common_area)
    if not common_area:
      continue
    print(f"Assigning common areas for model group {mg_id}...")
    common_area_filters: list | None = None
    if isinstance(common_area, list):
      common_area_filters = common_area
    print(f"common area filters = {common_area_filters}")
    df_univ = _assign_modal_model_group_to_common_area(df_univ, mg_id, common_area_filters)

  df_univ.to_parquet("out/look/tag-univ-1.parquet", engine="pyarrow")
  index_changed = ~old_model_group["model_group"].eq(df_univ["model_group"])
  rows_changed = df_univ[index_changed]
  print(f" --> {len(rows_changed)} parcels had their model group changed.")

  # TODO: fix this
  # Update sales for any rows that changed due to common area assignment
  # df_sales = combine_dfs(df_sales, rows_changed, df2_stomps=True, index="key")

  for mg_id in mg:
    entry = mg[mg_id]
    name = entry.get("name", mg_id)
    _filter = entry.get("filter", [])
    univ_index = resolve_filter(df_univ, _filter)
    sales_index = resolve_filter(df_sales_hydrated, _filter)
    if verbose:
      valid_sales_index = sales_index & df_sales_hydrated["valid_sale"].eq(True)
      improved_sales_index = sales_index & valid_sales_index & ~df_sales_hydrated["vacant_sale"].eq(True)
      vacant_sales_index = sales_index & valid_sales_index & df_sales_hydrated["vacant_sale"].eq(True)
      print(f"{name}")
      print(f"--> {univ_index.sum():,} parcels")
      print(f"--> {valid_sales_index.sum():,} sales")
      print(f"----> {improved_sales_index.sum():,} improved sales")
      print(f"----> {vacant_sales_index.sum():,} vacant sales")
  df_univ.loc[df_univ["model_group"].isna(), "model_group"] = "UNKNOWN"
  sup.set("universe", df_univ)
  sup.set("sales", df_sales)
  return sup


def _assign_modal_model_group_to_common_area(df_univ_in: gpd.GeoDataFrame, model_group_id: str, common_area_filters: list | None = None) -> gpd.GeoDataFrame:
  """
  Assign the modal model_group of parcels inside an enveloping "COMMON AREA" parcel to that parcel.

  Parameters:
      df_univ_in (gpd.GeoDataFrame): GeoDataFrame containing all parcels.
      model_group_id (str): Target model group identifier.
      common_area_filters (list, optional): Filters to further select common area parcels.

  Returns:
      gpd.GeoDataFrame: Modified GeoDataFrame with updated model_group for COMMON AREA parcels.
  """
  df_univ = df_univ_in.copy()

  # Ensure geometry column is set
  if df_univ.geometry.name is None:
    raise ValueError("GeoDataFrame must have a geometry column.")

  # Reduce df_univ to ONLY those parcels that have holes in them:
  df = identify_parcels_with_holes(df_univ)

  print(f" {len(df)} parcels with holes found.")
  df.to_parquet("out/look/common_area-0-holes.parquet", engine="pyarrow")
  df["has_holes"] = True

  if common_area_filters is not None:
    df_extra = select_filter(df_univ, common_area_filters).copy()
    df_extra["is_common_area"] = True
    print(f" {len(df_extra)} extra parcels found.")
    df = pd.concat([df, df_extra], ignore_index=True)
    # drop duplicate keys:
    df = df.drop_duplicates(subset="key")

  print(f" {len(df)} potential COMMON AREA parcels found.")
  df.to_parquet("out/look/common_area-1-common_area.parquet", engine="pyarrow")

  print(f"Assigning modal model_group to {len(df)}/{len(df_univ_in)} potential parcels...")

  df["modal_tagged"] = None

  # Iterate over COMMON AREA parcels
  for idx, row in df.iterrows():
    # Get the envelope of the COMMON AREA parcel
    common_area_geom = row.geometry
    common_area_gs = gpd.GeoSeries([common_area_geom], crs=df.crs)
    common_area_envelope_geom = common_area_geom.envelope
    common_area_envelope_gs = gpd.GeoSeries([common_area_envelope_geom], crs=df.crs)

    geom = common_area_geom.buffer(0)
    if geom.geom_type == "Polygon":
      outer_polygon = Polygon(geom.exterior)
    elif geom.geom_type == "MultiPolygon":
      outer_polygons = [Polygon(poly.exterior) for poly in geom.geoms]
      outer_polygon = unary_union(outer_polygons)
    else:
      raise ValueError("Geometry must be a Polygon or MultiPolygon")
    #outer_polygon_gs = gpd.GeoSeries([outer_polygon], crs=df.crs)

    # Find parcels wholly inside the COMMON AREA envelope
    inside_parcels = df_univ_in[df_univ_in.geometry.within(common_area_envelope_geom)].copy()

    # buffer 0 on inside parcel geometry
    inside_parcels["geometry"] = inside_parcels["geometry"].apply(lambda g: g.buffer(0))

    count1 = len(inside_parcels)

    # Exclude the COMMON AREA parcel itself (if it is in df_univ)
    inside_parcels = inside_parcels[~inside_parcels.geometry.apply(lambda g: g.equals(common_area_geom))]
    count2 = len(inside_parcels)

    # Optionally use a tiny negative buffer to avoid boundary issues

    # Exclude parcels that are not wholly inside the COMMON AREA parcel (not just the envelope bounding box):
    if isinstance(outer_polygon, np.ndarray):
      if outer_polygon.size == 1:
        outer_polygon = outer_polygon[0]
      else:
        # If there are multiple elements, combine them into one geometry
        outer_polygon = unary_union(list(outer_polygon))
      print("outer_polygon type:", type(outer_polygon))
    inside_parcels = inside_parcels[inside_parcels.geometry.centroid.within(outer_polygon)]
    count3 = len(inside_parcels)

    print(f" {idx} --> {count1} parcels inside the envelope, {count2} after excluding the COMMON AREA, {count3} after excluding those not wholly inside the COMMON AREA")

    # If it's empty, continue:
    if inside_parcels.empty:
      continue

    # Check that at least one of the inside_parcels matches the target model_group_id, otherwise continue:
    if not inside_parcels["model_group"].eq(model_group_id).any():
      continue

    # Determine the modal model_group value
    modal_model_group = inside_parcels["model_group"].value_counts().index[0]
    if modal_model_group is not None and modal_model_group != "":
      print(f" {idx} --> modal model group = {modal_model_group} for {len(inside_parcels)} inside parcels")
      # Apply the modal model_group to the COMMON AREA parcel
      df.at[idx, "model_group"] = modal_model_group
      df.at[idx, "modal_tagged"] = True
    else:
      print(f" {idx} --> XXX modal model group is {modal_model_group} for {len(inside_parcels)} inside parcels")

  df.to_parquet("out/look/common_area-2-tagged.parquet", engine="pyarrow")
  df_return = df_univ_in.copy()
  # Update and return df_univ
  df_return = combine_dfs(df_return, df[["key", "model_group"]], df2_stomps=True, index="key")
  df_return.to_parquet("out/look/common_area-3-return.parquet", engine="pyarrow")
  return df_return
