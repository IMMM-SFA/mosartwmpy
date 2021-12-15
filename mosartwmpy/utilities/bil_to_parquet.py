import click
import geopandas as gpd
import json
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box
import xarray as xr

@click.command()
@click.option(
    '--grid-path',
    default='input/domains/mosart.nc',
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
    default=['input/elevation/na_dem_30s_bil/na_dem_30s.bil', 'input/elevation/ca_dem_30s_bil/ca_dem_30s.bil'],
    multiple=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    prompt='What is the path to the .bil elevation file?',
    help="""Path to one or more .bil elevation file(s).""",
)
@click.option(
    '--parquet-elevation-path',
    default='input/elevation/na_dem_30s.parquet',
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
    """Convert one or more bil file(s) into a parquet file."""

    domain = xr.open_dataset(grid_path)
    grid_resolution = domain[grid_latitude_key][1] - domain[grid_latitude_key][0]
    ID = domain['ID'].values.flatten()

    # Combine all input bil files.
    merged_bil = None
    for bil in bil_elevation_path:
        if merged_bil is None:
            merged_bil = rasterio.open(bil)
            continue

        bil = rasterio.open(bil)
        merged_bil, transform = merge([bil, merged_bil])
        merged_bil = return_in_memory(merged_bil, bil.crs, transform)

    merged_bil = avg_resample(merged_bil, grid_resolution)
    merged_bil = crop_to_domain(merged_bil, domain, grid_longitude_key, grid_latitude_key, grid_resolution)

    df = pd.DataFrame(merged_bil.read(1).flatten())
    df.columns = df.columns.astype(str)
    df.to_parquet(parquet_elevation_path)

def avg_resample(bil, grid_resolution):
    scale_factor = bil.res[0] / grid_resolution

    avg_sampled_bil = bil.read(
                            out_shape=(
                                bil.count,
                                int(bil.height * scale_factor),
                                int(bil.width * scale_factor)
                            ),
                            resampling=rasterio.enums.Resampling.average
    )
    transform = bil.transform * bil.transform.scale(
        (bil.width / avg_sampled_bil.shape[-1]),
        (bil.height / avg_sampled_bil.shape[-2])
    )
    return return_in_memory(avg_sampled_bil, bil.crs, transform)

def crop_to_domain(bil, domain, grid_latitude_key, grid_longitude_key, grid_resolution):
    xmin, ymin, xmax, ymax = domain[grid_latitude_key].min().min().item(0), domain[grid_longitude_key].min().min().item(0), domain[grid_latitude_key].max().max().item(0), domain[grid_longitude_key].max().max().item(0)
    bbox = box(xmin, ymin, xmax + grid_resolution, ymax + grid_resolution)

    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=bil.crs)
    coords = [json.loads(geo.to_json())['features'][0]['geometry']]
    cropped, transform = mask(dataset=bil, shapes=coords, crop=True, nodata=-999)

    return return_in_memory(cropped, bil.crs, transform)

def return_in_memory(array, crs, transform):
    memfile = MemoryFile()
    dataset = memfile.open(driver='GTiff', height=array.shape[-2], width=array.shape[-1], count=1, crs=crs, transform=transform, dtype=array.dtype)
    dataset.write(array)
    return dataset

if __name__ == '__main__':
    bil_to_parquet()
