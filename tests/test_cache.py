import os

import geopandas
import pandas as pd

from openavmkit.synthetic.basic import generate_basic
from openavmkit.utilities.assertions import dicts_are_equal, dfs_are_equal
from openavmkit.utilities.cache import check_cache, write_cache, read_cache


def test_cache():

  signature = {
    'id': '12345'
  }

  synthetic = generate_basic(100)
  gdf = synthetic.df_universe
  df = gdf.drop(columns=["geometry"])
  df = pd.DataFrame(df)

  trials = [
    {
      "extension": "txt",
      "filetype": "str",
      "payload": "I am the very model of a modern Major General, I've information vegetable, animal, and mineral. I know the kings of England, and I quote the fights historical, from Marathon to Waterloo, in order categorical. I'm very well acquainted too with matters mathematical, I understand equations, both the simple and quadratical. About binomial theorem I'm teeming with a lot o' news, with many cheerful facts about the square of the hypotenuse."
    },
    {
      "extension": "json",
      "filetype": "dict",
      "payload":{
        'a': [i for i in range(0, 3)],
        'b': [i*2 for i in range(0, 3)],
        'c': [i*3 for i in range(0, 3)],
      }
    },
    {
      "extension": "parquet",
      "filetype": "df",
      "payload": df
    },
    {
      "extension": "parquet",
      "filetype": "gdf",
      "payload": gdf
    }
  ]

  for trial in trials:

    ext = trial["extension"]
    filetype = trial["filetype"]
    payload: dict | str | pd.DataFrame | geopandas.GeoDataFrame = trial["payload"]

    os.makedirs("cache", exist_ok=True)
    # clear every FILE in the cache:
    for file in os.listdir("cache"):
      file_path = os.path.join("cache", file)
      if os.path.isfile(file_path):
        os.remove(file_path)

    is_cached = check_cache("test_cache", signature, filetype)

    assert is_cached == False

    write_cache("test_cache", payload, signature, filetype)

    # check if the cache file exists:
    assert os.path.exists(f"cache/test_cache.{ext}")

    # check if the cache file is not empty:
    assert os.path.getsize(f"cache/test_cache.{ext}") > 0

    # check if the cache file is equal to the payload:
    if filetype == "dict":
      with open("cache/test_cache.json", "r") as f:
        cache = f.read()
        assert str(cache).replace("\"", "'") == str(payload).replace("\"", "'")
    elif filetype == "str":
      with open("cache/test_cache.txt", "r") as f:
        cache = f.read()
        assert str(cache) == str(payload)
    elif filetype == "df" or filetype == "gdf":
      cache = pd.read_parquet("cache/test_cache.parquet")
      if filetype == "gdf":
        cache["geometry"] = geopandas.GeoSeries.from_wkb(cache["geometry"])
      assert dfs_are_equal(cache, payload, "key")

    is_cached = check_cache("test_cache", signature, filetype)

    assert is_cached == True

    # read the cache file:
    cached_file = read_cache("test_cache", filetype)

    # check if the cached file is equal to the payload:
    if filetype == "dict":
      assert dicts_are_equal(cached_file, payload)
    elif filetype == "str":
      assert str(cached_file) == str(payload)
    elif filetype == "df" or filetype == "gdf":
      assert dfs_are_equal(cached_file, payload, "key")

    # change the signature:
    dirty_signature = {
      "id": "54321"
    }

    # check the cache with the wrong signature:
    is_cached = check_cache("test_cache", dirty_signature, filetype)

    assert is_cached == False
