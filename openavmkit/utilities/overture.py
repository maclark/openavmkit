import os
import warnings
import geopandas as gpd
import pandas as pd
import tempfile
from pathlib import Path
import traceback
import subprocess
import sys
import shutil
import site

from openavmkit.utilities.geometry import get_crs
from openavmkit.utilities.timing import TimingData


class OvertureService:
    """Service for fetching and processing Overture building data."""
    
    def __init__(self, settings: dict):
        """Initialize the Overture service with settings."""
        self.settings = settings.get("overture", {})
        if not self.settings:
            warnings.warn("No Overture settings found in settings dictionary")
        self.cache_dir = "cache/overture"
        os.makedirs(self.cache_dir, exist_ok=True)


    def _find_overturemaps_executable(self):
        """Find the overturemaps executable in the current Python environment."""
        # First try the scripts directory of the current environment
        venv_scripts = os.path.join(sys.prefix, 'bin')
        overturemaps_path = os.path.join(venv_scripts, 'overturemaps')
        
        if os.path.exists(overturemaps_path):
            return overturemaps_path
            
        # Try user site-packages bin directory
        user_scripts = os.path.join(site.USER_BASE, 'bin')
        overturemaps_path = os.path.join(user_scripts, 'overturemaps')
        
        if os.path.exists(overturemaps_path):
            return overturemaps_path
            
        # Try which as a last resort
        overturemaps_path = shutil.which('overturemaps')
        if overturemaps_path:
            return overturemaps_path
            
        raise RuntimeError("Could not find overturemaps executable")

    def get_buildings(self, bbox, use_cache=True, verbose=False):
        """
        Fetch building data from Overture within the specified bounding box.
        
        Args:
            bbox: Tuple of (minx, miny, maxx, maxy) in WGS84
            use_cache: Whether to use cached data
            verbose: Whether to print detailed progress
            
        Returns:
            GeoDataFrame with building footprints
        """
        t = TimingData()
        try:
            if verbose:
                print(f"--> Current settings: {self.settings}")
                
            if not self.settings:
                if verbose:
                    print("--> No Overture settings found")
                return gpd.GeoDataFrame()
                
            if not self.settings.get("enabled", False):
                if verbose:
                    print("--> Overture service disabled in settings")
                return gpd.GeoDataFrame()

            if verbose:
                print(f"--> Bounding box: {bbox}")

            # Create cache signature
            cache_key = f"buildings_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.parquet")

            # Check cache
            if use_cache and os.path.exists(cache_path):
                if verbose:
                    print(f"--> Loading from cache: {cache_path}")
                return gpd.read_parquet(cache_path)

            if verbose:
                print("--> Fetching data from Overture...")

            try:
                # Create a temporary file for the GeoParquet output
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
                    temp_path = tmp_file.name

                # Format bbox string
                bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                
                if verbose:
                    print("--> Running overturemaps CLI...")

                # TODO: Replace this with a cross-platform solution that doesn't download an EXE
                # Find the overturemaps executable
                overturemaps_path = self._find_overturemaps_executable()
                if verbose:
                    print(f"--> Found overturemaps at: {overturemaps_path}")
                
                # Construct the command
                cmd = [
                    overturemaps_path,
                    "download",
                    "--bbox", bbox_str,
                    "-f", "geoparquet",
                    "--type", "building",
                    "-o", temp_path
                ]
                
                if verbose:
                    print(f"--> Command: {' '.join(cmd)}")
                
                # Run the command
                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join(sys.path)  # Ensure Python can find all packages
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,  # This will raise CalledProcessError if command fails
                    env=env
                )
                
                if verbose and result.stdout:
                    print(f"--> Command output: {result.stdout}")
                
                if verbose:
                    print("--> Reading GeoParquet file...")
                
                # Read the GeoParquet file
                gdf = gpd.read_parquet(temp_path)
                
                if verbose:
                    print(f"--> Found {len(gdf)} buildings")
                    if len(gdf) > 0:
                        print(f"--> Available columns: {gdf.columns.tolist()}")
                
                if not gdf.empty:
                    # Calculate footprint areas
                    t.start("area")
                    gdf["bldg_area_footprint_sqft"] = gdf.to_crs(gdf.estimate_utm_crs()).area * 10.764  # Convert m² to ft²
                    t.stop("area")
                    if verbose:
                        _t = t.get("area")
                        print(f"--> Calculated building footprint areas...({_t:.2f}s)")

                    if use_cache:
                        t.start("save")
                        gdf.to_parquet(cache_path)
                        t.stop("save")
                        if verbose:
                            _t = t.get("save")
                            print(f"--> Saving to cache: {cache_path}...({_t:.2f}s)")


                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass

                return gdf

            except subprocess.CalledProcessError as e:
                if verbose:
                    print(f"--> CLI command failed with error: {e.stderr}")
                    print(f"--> Command output: {e.stdout}")
                raise
            except Exception as e:
                if verbose:
                    print(f"--> Failed to fetch Overture data: {str(e)}")
                raise

        except Exception as e:
            if verbose:
                print(f"--> Error in get_buildings: {str(e)}")
                print(f"--> Traceback: {traceback.format_exc()}")
            warnings.warn(f"Failed to fetch Overture building data: {str(e)}\n{traceback.format_exc()}")
            return gpd.GeoDataFrame()

    def calculate_building_footprints(
        self,
        gdf: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        desired_units: str,
        field_name: str = "bldg_area_footprint_sqft",
        verbose: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Calculate building footprint areas for each parcel by intersecting with building geometries.
        
        Args:
            gdf: GeoDataFrame containing parcels
            buildings: GeoDataFrame containing building footprints
            desired_units: Units for area calculation (supported: "sqft", "sqm")
            verbose: Whether to print detailed progress
            
        Returns:
            GeoDataFrame with added building footprint areas
        """
        t = TimingData()
        if buildings.empty:
            if verbose:
                print("--> No buildings found, returning original GeoDataFrame")
            gdf["bldg_area_footprint_sqft"] = 0
            return gdf

        # Get appropriate unit conversion
        unit_mult = 1.0
        if desired_units == "sqft":
            unit_mult = 10.764  # Convert m² to sqft
        elif desired_units == "sqm":
            unit_mult = 1.0
        else:
            raise ValueError(f"Unsupported units: {desired_units}. Supported units are 'sqft' and 'sqm'.")

        t.start("crs")
        # Convert both to same CRS for spatial operations
        buildings = buildings.to_crs(gdf.crs)
        
        # Get appropriate CRS for area calculations
        area_crs = get_crs(gdf, "equal_area")

        # Project both datasets to equal area CRS for accurate area calculations
        buildings_projected = buildings.to_crs(area_crs)
        gdf_projected = gdf.to_crs(area_crs)
        t.stop("crs")

        if verbose:
            _t = t.get("crs")
            print(f"--> Projected to equal area CRS...({_t:.2f}s)")

        t.start("join")
        # Perform spatial join to find all building-parcel intersections
        joined = gpd.sjoin(gdf_projected, buildings_projected, how="left", predicate="intersects")
        t.stop("join")

        if verbose:
            _t = t.get("join")
            print(f"--> Calculated building footprint intersections with parcels...({_t:.2f}s)")

        if verbose:
            print(f"--> Found {len(joined)} potential building-parcel intersections")
        
        def calculate_intersection_area(row):
            try:
                parcel_geom = gdf_projected.loc[row.name, 'geometry']
                building_idx = row['index_right']
                if pd.isna(building_idx):
                    return 0.0
                building_geom = buildings_projected.loc[building_idx, 'geometry']
                if parcel_geom.intersects(building_geom):
                    intersection = parcel_geom.intersection(building_geom)
                    return intersection.area * unit_mult # Convert to desired units
                return 0.0
            except Exception as e:
                if verbose:
                    print(f"Warning: Error calculating intersection area: {e}")
                return 0.0

        t.start("calc_area")
        # TODO: Optimize this step using vectorized operations if possible
        # Calculate intersection areas
        joined[field_name] = joined.apply(calculate_intersection_area, axis=1)
        t.stop("calc_area")

        # Calculate actual intersection areas
        if verbose:
            _t = t.get("calc_area")
            print(f"--> Calculating precise intersection areas...({_t:.2f}s)")

        # Aggregate total building footprint area per parcel
        t.start("agg")
        agg = joined.groupby("key")[field_name].sum().reset_index()
        t.stop("agg")

        if verbose:
            _t = t.get("agg")
            print(f"--> Aggregating building footprint areas...({_t:.2f}s)")

        t.start("finish")
        # Merge back to original dataframe
        gdf = gdf.merge(agg, on="key", how="left", suffixes=("", "_agg"))

        if f"{field_name}_agg" in gdf.columns:
            # If the original field name existed, then we will stomp with non-null values from the calculated field
            gdf.loc[~gdf[f"{field_name}_agg"].isna(), field_name] = gdf[f"{field_name}_agg"]
            gdf.drop(columns=[f"{field_name}_agg"], inplace=True)

        # Fill NaN values with 0 (parcels with no buildings)
        gdf[field_name] = gdf[field_name].fillna(0)
        t.stop("finish")

        if verbose:
            _t = t.get("finish")
            print(f"--> Finished up...({_t:.2f}s)")
            print(f"--> Added building footprint areas to {len(agg)} parcels")
            print(f"--> Total building footprint area: {gdf[field_name].sum():,.0f} sqft")
            print(f"--> Average building footprint area: {gdf[field_name].mean():,.0f} sqft")
            print(f"--> Number of parcels with buildings: {(gdf[field_name] > 0).sum():,}")
            
        return gdf

def init_service_overture(settings: dict) -> OvertureService:
    """Initialize the Overture service."""
    return OvertureService(settings) 