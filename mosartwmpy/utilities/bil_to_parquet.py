import click
import numpy as np
import geopandas as gpd
from matplotlib import pyplot
import matplotlib as plt
import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
import pycrs
import rasterio
from rasterio.mask import mask
# from rasterio.warp import reproject, Resampling
from shapely.geometry import box
import sys
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
        upscale_elevation=False,
        grid_longitude_key='lon',
        grid_latitude_key='lat',
):
    """Convert a bil file into a parquet file."""

    # read grid domain file
    domain = xr.open_dataset(grid_path)
    longitude, latitude = np.meshgrid(domain[grid_longitude_key], domain[grid_latitude_key])
    grid_resolution = latitude[1][0] - latitude[0][0]
    longitude = longitude.flatten()
    latitude = latitude.flatten()
    ID = domain['ID'].to_numpy().flatten()
    
    # import bil elevation file and trim to domain
    bil = rasterio.open(bil_elevation_path)
    bil = cropToDomain(bil, domain, grid_longitude_key, grid_latitude_key, grid_resolution,  bil_elevation_path[:-4] + '_cropped.bil')

    data = bil.read()[0]  # this would be the raw elevation data with no lat/lon info
    bounds = bil.bounds  # this is the bounding box, coordinates
    res = bil.res  # this is the resolution
    shape = bil.shape  # this is the array size

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
    # scale image transform
    downsampled_transform = bil.transform * bil.transform.scale(
        (bil.width / avg_downsampled_bil.shape[-1]),
        (bil.height / avg_downsampled_bil.shape[-2])
    )

    min_downsampled_bil = bil.read(
                            out_shape=(
                                bil.count,
                                int(bil.height * scale_factor),
                                int(bil.width * scale_factor)
                            ),
                            resampling=rasterio.enums.Resampling.min
    )

    # pyplot.imshow(bil.read(1))
    # pyplot.show()

    # TODO: return as a parquet file




def cropToDomain(bil, domain, grid_latitude_key, grid_longitude_key, grid_resolution, cropped_output_path):
    xmin, ymin, xmax, ymax = domain[grid_latitude_key].min().min().item(0), domain[grid_longitude_key].min().min().item(0), domain[grid_latitude_key].max().max().item(0), domain[grid_longitude_key].max().max().item(0)
    bbox = box(xmin, ymin, xmax + grid_resolution, ymax + grid_resolution)

    if bbox == bil.bounds:
        return bil
    # elif rasterio.coords.disjoint_bounds(bbox, bil.bounds):
    #     sys.exit("Land areas do not overlap.")

    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=bil.crs)
    coords = getFeatures(geo)
    out_img, out_transform = mask(dataset=bil, shapes=coords, crop=True)
    out_meta = bil.meta.copy()
    epsg_code = int(bil.crs.data['init'][5:])
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform,
                     "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()})
    with rasterio.open(cropped_output_path, "w", **out_meta) as dest:
        dest.write(out_img)
    
    return rasterio.open(cropped_output_path)

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

bil_to_parquet()
