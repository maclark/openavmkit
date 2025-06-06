import os
import json
import pandas as pd
import geopandas as gpd

from openavmkit.utilities.assertions import objects_are_equal, dicts_are_equal, dfs_are_equal
from openavmkit.utilities.geometry import ensure_geometries


def write_cache(
    filename: str,
    payload: dict | str | pd.DataFrame | gpd.GeoDataFrame,
    signature: dict | str,
    filetype: str
):
  extension = _get_extension(filetype)
  path = f"cache/{filename}.{extension}"
  base_path = os.path.dirname(path)
  os.makedirs(base_path, exist_ok=True)
  with open(path, "w") as file:
    if filetype == "dict":
      json.dump(payload, file)
    elif filetype == "str":
      file.write(payload)
    elif filetype == "df":
      if isinstance(payload, pd.DataFrame):
        if isinstance(payload, gpd.GeoDataFrame):
          payload.to_parquet(path, engine="pyarrow")
        else:
          payload.to_parquet(path)
      else:
        raise TypeError("Payload must be a DataFrame for df type.")

  if type(signature) is dict:
    sig_ext = "json"
  elif type(signature) is str:
    sig_ext = "txt"
  else:
    raise TypeError(f"Unsupported type for signature value: {type(signature)} sig = {signature}")

  signature_path = f"cache/{filename}.signature.{sig_ext}"
  with open(signature_path, "w") as file:
    if sig_ext == "json":
      json.dump(signature, file)
    else:
      file.write(signature)


def read_cache(
    filename: str,
    filetype: str
):
  extension = _get_extension(filetype)
  path = f"cache/{filename}.{extension}"
  if os.path.exists(path):
    with open(path, "r") as file:
      if filetype == "dict":
        return json.load(file)
      elif filetype == "str":
        return file.read()
      elif filetype == "df":
        import pandas as pd
        try:
          df = gpd.read_parquet(path)
          if "geometry" in df:
            df = gpd.GeoDataFrame(df, geometry="geometry")
            ensure_geometries(df, "geometry", df.crs)
        except ValueError:
          df = pd.read_parquet(path)
        return df
  return None


def check_cache(
    filename: str,
    signature: dict | str,
    filetype: str
):
  ext = _get_extension(filetype)
  path = f"cache/{filename}"
  match = _match_signature(path, signature)
  if match:
    path_exists = os.path.exists(f"{path}.{ext}")
    return path_exists
  return False


def clear_cache(
    filename: str,
    filetype: str
):
  ext = _get_extension(filetype)
  path = f"cache/{filename}"
  if os.path.exists(f"{path}.{ext}"):
    os.remove(f"{path}.{ext}")
  if os.path.exists(f"{path}.signature.json"):
    os.remove(f"{path}.signature.json")


def write_cached_df(
    df_orig: pd.DataFrame,
    df_new: pd.DataFrame,
    filename: str,
    key: str = "key",
    extra_signature: dict | str = None
)-> pd.DataFrame | None:

  orig_cols = set(df_orig.columns)
  new_cols  = [c for c in df_new.columns if c not in orig_cols]
  common    = [c for c in df_new.columns if c in orig_cols]

  modified = []
  for c in common:
    # nan‑safe check: True if ANY value differs
    neq = ~(df_new[c].eq(df_orig[c]) | (df_new[c].isna() & df_orig[c].isna()))
    if neq.any():
      modified.append(c)

  changed_cols = new_cols + modified
  if not changed_cols:
    # nothing new or modified → no cache update needed
    return

  df_diff = df_new[[key]+changed_cols].copy()

  signature = _get_df_signature(df_orig, extra_signature)

  df_type = "df"

  write_cache(filename, df_diff, signature, df_type)

  df_cached = get_cached_df(df_orig, filename, key, extra_signature)

  assert dfs_are_equal(df_new, df_cached, allow_weak=True)

  return df_cached

def get_cached_df(
    df: pd.DataFrame,
    filename: str,
    key: str = "key",
    extra_signature: dict | str = None
)->pd.DataFrame | gpd.GeoDataFrame | None:
  signature = _get_df_signature(df, extra_signature)

  if check_cache(filename, signature, "df"):
    df_diff = read_cache(filename, "df")
    if df_diff is None or df_diff.empty:
      return None

    df_diff[key] = df_diff[key].astype(df[key].dtype)

    cols_to_replace = [c for c in df_diff.columns if c != key]
    df_base = df.drop(columns=cols_to_replace, errors="ignore")

    df_merged = df_base.merge(df_diff, how="left", on=key)

    if isinstance(df_diff, gpd.GeoDataFrame):
      df_merged = gpd.GeoDataFrame(df_merged, geometry="geometry")
      df_merged = ensure_geometries(df_merged, "geometry", df_diff.crs)

    return df_merged

  return None


def _get_df_signature(df: pd.DataFrame, extra: dict | str = None):
  sorted_columns = sorted(df.columns)
  signature = {
    "num_rows": len(df),
    "num_columns": len(df.columns),
    "columns": sorted_columns,
    "checksum": _cheap_checksum(df)
  }
  if extra is not None:
    signature["extra"] = extra
  return signature


def _cheap_checksum(df: pd.DataFrame):
  checksum = {}
  return checksum
  # for col in df.columns:
  #   # if it's geometry:
  #   # if it's numeric:
  #   if pd.api.types.is_numeric_dtype(df[col]):
  #     checksum[col] = float(df[col].sum())
  #   elif col == "geometry":
  #     # just note how many geometry rows are not null:
  #     checksum[col] = float((~df[col].isna()).sum())
  #   else:
  #     try:
  #       checksum[col] = str(df[col].value_counts())
  #     except TypeError:
  #       checksum[col] = float(df[col].apply(lambda x: str(x).encode("utf-8")).sum())
  # return checksum

def _match_signature(
    filename: str,
    signature: dict | str
)->bool:
  if type(signature) is dict:
    sig_ext = "json"
  elif type(signature) is str:
    sig_ext = "txt"
  else:
    raise TypeError(f"Unsupported type for signature value: {type(signature)}")
  sig_file = f"{filename}.signature.{sig_ext}"
  match = False
  if os.path.exists(sig_file):
    if sig_ext == "json":
      with open(sig_file, "r") as file:
        cache_signature = json.load(file)
      match = dicts_are_equal(signature, cache_signature)
    else:
      with open(sig_file, "r") as file:
        cache_signature = file.read()
      match = signature == cache_signature
  return match


def _get_extension(filetype:str):
  if filetype == "dict":
    return "json"
  elif filetype == "str":
    return "txt"
  elif filetype == "df":
    return "parquet"
  elif filetype == "json":
    raise ValueError(f"Filetype 'json' is unsupported, did you mean 'dict'?")
  elif filetype == "txt" or filetype == "text":
    raise ValueError(f"Filetype '{filetype}' is unsupported, did you mean 'str'?")
  elif filetype == "parquet":
    raise ValueError(f"Filetype 'parquet' is ambiguous: please use 'df' instead")
  raise ValueError(f"Unsupported filetype: '{filetype}'")
