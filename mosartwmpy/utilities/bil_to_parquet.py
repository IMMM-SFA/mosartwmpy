import click
import numpy as np
import geopandas as gpd
import json
from matplotlib import pyplot
import matplotlib as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from shapely.geometry import box
import xarray as xr

@click.command()
@click.option(
    '--grid-path',
    default='../../input/domains/mosart.nc',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    prompt='What is the path to the domain/grid file?',
    help="""Path to the domain/grid file representing where to place GRanD dams."""
)
@click.option(
    '--bil-elevation-path',
    default='../../input/elevation/na_dem_30s_bil/na_dem_30s.bil',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    prompt='What is the path to the .bil elevation file?',
    help="""Path to the .bil elevation file.""",
)
@click.option(
    '--parquet-elevation-path',
    default='../../input/elevation/na_dem_30s.parquet',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    prompt='What is the output path to the .parquet elevation file?',
    help="""Path to output the .parquet elevation file to.""",
)

def bil_to_parquet(
        grid_path,
        bil_elevation_path,
        parquet_elevation_path,
        grid_longitude_key='lon',
        grid_latitude_key='lat',
):
    """Convert a bil file into a parquet file."""

    domain = xr.open_dataset(grid_path)
    grid_resolution = domain[grid_latitude_key][1] - domain[grid_latitude_key][0]
    ID = domain['ID'].to_numpy().flatten()
    
    # Import bil elevation file and trim to domain.
    bil = rasterio.open(bil_elevation_path)
    bil = cropToDomain(bil, domain, grid_longitude_key, grid_latitude_key, grid_resolution,  bil_elevation_path[:-4] + '_cropped.bil')

    # Resample data to same resolution as grid.
    scale_factor = bil.res[0]/grid_resolution
    avg_downsampled_bil = bil.read(
                            out_shape=(
                                bil.count,
                                int(bil.height * scale_factor),
                                int(bil.width * scale_factor)
                            ),
                            resampling=rasterio.enums.Resampling.average
    )

    # Write as parquet file.
    df = pd.DataFrame(avg_downsampled_bil.flatten(), ID)
    df.columns = df.columns.astype(str)
    df.to_parquet(parquet_elevation_path)


def cropToDomain(bil, domain, grid_latitude_key, grid_longitude_key, grid_resolution, cropped_output_path):
    xmin, ymin, xmax, ymax = domain[grid_latitude_key].min().min().item(0), domain[grid_longitude_key].min().min().item(0), domain[grid_latitude_key].max().max().item(0), domain[grid_longitude_key].max().max().item(0)
    bbox = box(xmin, ymin, xmax + grid_resolution, ymax + grid_resolution)

    if bbox == bil.bounds:
        return bil

    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=bil.crs)
    coords = [json.loads(geo.to_json())['features'][0]['geometry']]
    out_img, out_transform = mask(dataset=bil, shapes=coords, crop=True)
    out_meta = bil.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform,
                     "crs": bil.crs})
    
    with MemoryFile() as memfile:
        with memfile.open(**out_meta) as dataset: # Open as DatasetWriter
            dataset.write(out_img)
            del out_img
        return memfile.open()

if __name__ == '__main__':
    bil_to_parquet()
