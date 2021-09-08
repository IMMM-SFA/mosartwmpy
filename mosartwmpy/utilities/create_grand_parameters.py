import click

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import xarray as xr
from scipy.spatial import KDTree
import contextily as ctx
import matplotlib.pyplot as plt


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
    '--grand-path',
    default='input/reservoirs/GRanD_reservoirs_v1_3.shp',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    prompt='What is the path to the GRanD reservoir file?',
    help="""Path to the GRanD dam file."""
)
@click.option(
    '--elevation-path',
    default='input/reservoirs/hydroshed_upscaled_elevation.parquet',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    prompt='What is the path to the elevation file?',
    help="""Path to the elevation file.""",
)
@click.option(
    '--istarf-path',
    default='input/reservoirs/ISTARF-CONUS.csv',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    prompt='What is the path to the ISTARF parameter file?',
    help="""Path to the ISTARF file providing harmonic parameters for reservoir releases.""",
)
@click.option(
    '--demand-path',
    default='input/demand/RCP8.5_GCAM_water_demand_1980_2011.nc',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    prompt='What is the path to the demand file for which to find the monthly average demand?',
    help="""Path to the demand file for finding monthly average demand."""
)
@click.option(
    '--flow-path',
    default='output/monthly_mean_streamflow',
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=False,
        resolve_path=True,
    ),
    prompt='What is the path to a folder containing mosartwmpy output for which to find the monthly average flow?',
    help="""Path to the directory containing mosartwmpy output for finding monthly average flow."""
)
@click.option(
    '--reservoir-output-path',
    default='./grand_reservoir_parameters.nc',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
    ),
    prompt='Where should the reservoir parameter file be written?',
    help="""The path to which the reservoir parameters should be written."""
)
@click.option(
    '--dependency-output-path',
    default='./grand_dependency_database.parquet',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
    ),
    prompt='Where should the reservoir dependency database be written?',
    help="""The path to which the dependency database should be written."""
)
@click.option(
    '--average-monthly-demand-output-path',
    default='./grand_average_monthly_demand.parquet',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
    ),
    prompt='Where should the reservoir average monthly demand be written?',
    help="""The path to which the average monthly demand should be written."""
)
@click.option(
    '--average-monthly-flow-output-path',
    default='./grand_average_monthly_flow.parquet',
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
    ),
    prompt='Where should the reservoir average monthly flow be written?',
    help="""The path to which the average monthly flow should be written."""
)
def create_grand_parameters(
    grid_path,
    grand_path,
    elevation_path,
    istarf_path,
    demand_path,
    flow_path,
    reservoir_output_path,
    dependency_output_path,
    average_monthly_demand_output_path,
    average_monthly_flow_output_path,
    upscale_elevation=False,
    grid_longitude_key='lon',
    grid_latitude_key='lat',
    grid_downstream_key='dnID',
    istarf_grand_id_key='GRanD_ID',
    demand_key='totalDemand',
    demand_time_key='time',
    flow_key='channel_inflow',
    flow_time_key='time',
    elevation_key='hydroshed_average_elevation',
    elevation_upscale_cells=225,
    dependency_radius_meters=200000,
    dam_move_threshold=0.75,
    istarf_key_map=dict(
        GRanD_MEANFLOW_CUMECS='grand_meanflow_cumecs',
        Obs_MEANFLOW_CUMECS='observed_meanflow_cumecs',
        fit='fit',
        match='match',
        NORhi_alpha='upper_alpha',
        NORhi_beta='upper_beta',
        NORhi_max='upper_max',
        NORhi_min='upper_min',
        NORhi_mu='upper_mu',
        NORlo_alpha='lower_alpha',
        NORlo_beta='lower_beta',
        NORlo_max='lower_max',
        NORlo_min='lower_min',
        NORlo_mu='lower_mu',
        Release_alpha1='release_alpha_one',
        Release_alpha2='release_alpha_two',
        Release_beta1='release_beta_one',
        Release_beta2='release_beta_two',
        Release_c='release_c',
        Release_max='release_max',
        Release_min='release_min',
        Release_p1='release_p_one',
        Release_p2='release_p_two',
    ),
):
    """Create a reservoir parameter file based on an input grid file, an input GRanD dams file, and an input
       ISTARF parameter file."""

    # read the files
    domain = xr.open_dataset(grid_path)
    grand = gpd.read_file(grand_path)
    istarf = pd.read_csv(istarf_path)
    if upscale_elevation:
        elevation = gpd.read_parquet(elevation_path)
    else:
        elevation = pd.read_parquet(elevation_path)
    elevation.loc[elevation[elevation[elevation_key] < 0].index, elevation_key] = np.nan

    # create the grid as a geopandas dataframe
    longitude, latitude = np.meshgrid(domain[grid_longitude_key], domain[grid_latitude_key])
    longitude = longitude.flatten()
    latitude = latitude.flatten()
    grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(longitude, latitude))
    grid['GRID_CELL_INDEX'] = grid.index
    grid['DOWNSTREAM_INDEX'] = domain[grid_downstream_key].values.flatten().astype(int) - 1

    # create a dam geometry point column
    grand['DAM_POINT'] = grand.apply(lambda dam: Point(dam.LONG_DD, dam.LAT_DD), axis=1)
    grand['RESERVOIR_CENTROID'] = grand.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
    grand['REPRESENTATIVE_POINT'] = grand.to_crs(epsg=3857).representative_point().to_crs(epsg=4326)

    # calculate grid cell size in meters (because of reprojection)
    x_spacing = abs(domain[grid_longitude_key][0] - domain[grid_longitude_key][1])
    y_spacing = abs(domain[grid_latitude_key][0] - domain[grid_latitude_key][1])
    domain.close()

    # remove grand dams that do not appear in the ISTARF database
    grand = grand[grand.GRAND_ID.isin(istarf[istarf_grand_id_key])].set_geometry('DAM_POINT').copy().reset_index()
    click.echo(f'GRAND dams appearing in ISTARF data: {len(grand)}')

    # project the grid to web mercator
    grid = grid.set_crs(epsg=4326).to_crs(epsg=3857)

    # create the kdtree to search nearest neighbors, in a reasonable reference frame
    kdtree = KDTree(np.array(list(grid.geometry.apply(lambda p: (p.x, p.y)))))

    # find nearest domain index for each GRanD dam
    grand['GRID_CELL_INDEX'] = grand.set_crs(epsg=4326).to_crs(epsg=3857).geometry.apply(
        lambda p: kdtree.query((p.x, p.y), k=1)[1]
    )

    if upscale_elevation:
        # create the kdtree to search nearest elevations, in a reasonable reference frame
        elevation_kdtree = KDTree(
            np.array(list(elevation.set_crs(epsg=4326).to_crs(epsg=3857).geometry.apply(lambda p: (p.x, p.y))))
        )

        # set the grid cell elevations
        grid['ELEVATION'] = grid.geometry.apply(
            lambda p: elevation.iloc[elevation_kdtree.query(
                (p.x, p.y),
                k=elevation_upscale_cells,
            )[1]][elevation_key].mean()
        )

    else:
        grid['ELEVATION'] = elevation[elevation_key].values

    # set the outlet indices
    outlet_index = np.full(len(grid), -1)
    for i in np.arange(len(grid)):
        if grid['DOWNSTREAM_INDEX'].iloc[i] >= 0:
            j = i
            while True:
                k = grid['DOWNSTREAM_INDEX'].iloc[j]
                if (k >= 0) and (k != j):
                    j = k
                else:
                    outlet_index[i] = j
                    break
        else:
            outlet_index[i] = i
    grid['OUTLET_INDEX'] = outlet_index

    # for dams appearing in the same grid cell, check if the reservoir centroid would place the dam elsewhere
    grouped = grand.set_geometry('RESERVOIR_CENTROID')\
        .set_crs(epsg=4326).to_crs(epsg=3857).groupby('GRID_CELL_INDEX', as_index=False)
    for _, group in grouped:
        for i, row in enumerate(group.itertuples()):
            if len(group) > 1:
                # only allow a move if dam coordinate is close to cell border
                cell_centroid = grid.iloc[[row.GRID_CELL_INDEX]].to_crs(epsg=4326).geometry.values[0]
                if ((abs(cell_centroid.x - row.DAM_POINT.x) / (x_spacing / 2)) > dam_move_threshold) or \
                        ((abs(cell_centroid.y - row.DAM_POINT.y) / (y_spacing / 2)) > dam_move_threshold):
                    centroid_nearest_index = kdtree.query(
                        (row.RESERVOIR_CENTROID.x, row.RESERVOIR_CENTROID.y), k=1)[1]
                    if centroid_nearest_index != row.GRID_CELL_INDEX:
                        grand.at[row.Index, 'GRID_CELL_INDEX'] = centroid_nearest_index
    # finally, remove remaining duplicate dams, preferring largest capacity, then largest drainage area, then latest id
    filtered = grand.sort_values(['CAP_MCM', 'AREA_SKM', 'GRAND_ID'], ascending=False).groupby(
        'GRID_CELL_INDEX', as_index=False, group_keys=False).first()

    click.echo(f'GRAND dams remaining after deduplication: {len(filtered.index)}')

    # find the grid cells dependent on each dam
    dependent_cell_indices = []
    for i in np.arange(len(filtered)):
        dam_cell = grid.iloc[filtered.iloc[i]['GRID_CELL_INDEX']]
        cell_indices = set()
        # follow the river downstream and find the points within radius
        j = filtered.iloc[i]['GRID_CELL_INDEX']
        while True:
            k = grid.iloc[j]['DOWNSTREAM_INDEX']
            if (k >= 0) and (k != j):
                point = grid.iloc[k]
                indices = kdtree.query_ball_point((point.geometry.x, point.geometry.y), r=dependency_radius_meters)
                cell_indices.update(indices)
                j = k
            else:
                break
        cell_indices = [i for i in cell_indices if i < len(grid)]
        # restrict to cells below dam elev and with same outlet
        dependent_cells = grid.iloc[cell_indices]
        dependent_cells = dependent_cells[dependent_cells['ELEVATION'] >= 0]
        dependent_cells = dependent_cells[dependent_cells['ELEVATION'] < dam_cell['ELEVATION']]
        dependent_cells = dependent_cells[dependent_cells['OUTLET_INDEX'] == dam_cell['OUTLET_INDEX']]
        dependent_cells = dependent_cells.index.values
        dependent_cell_indices.append(dependent_cells[~np.isnan(dependent_cells)].astype(int))
    filtered['DEPENDENT_CELL_INDICES'] = dependent_cell_indices

    filtered.index = filtered.GRAND_ID.values
    filtered = filtered.sort_index()

    dependency_database = filtered[['GRAND_ID', 'DEPENDENT_CELL_INDICES']].explode(column='DEPENDENT_CELL_INDICES').rename(
        columns={'DEPENDENT_CELL_INDICES': 'DEPENDENT_CELL_INDEX'}).dropna().sort_values('GRAND_ID').reset_index(
        drop=True)

    # find the monthly average demand on each dam
    # each grid cell assigns it's demand in proportion to the capacity of each reservoir it depends on
    demand = xr.open_dataset(demand_path)[[demand_key]].load()
    mean_monthly_demand = np.nan_to_num(demand[demand_key].groupby(f'{demand_time_key}.month').mean().compute())
    demand.close()
    demand_rows = []
    dependent_reservoirs_by_cell = dependency_database.groupby('DEPENDENT_CELL_INDEX')['GRAND_ID'].apply(list)
    for m in np.arange(12):
        mean_demand_m = mean_monthly_demand[m, :, :].flatten()
        demand_allocations = pd.Series(index=filtered.index, data=0.0)
        for i in dependent_reservoirs_by_cell.index:
            grand_ids = dependent_reservoirs_by_cell.loc[i]
            capacities = filtered.loc[grand_ids].CAP_MCM
            demand_allocations.loc[grand_ids] = demand_allocations.loc[grand_ids] + \
                                                mean_demand_m[i] * capacities.values / capacities.sum()
        for i in np.arange(filtered.index.size):
            demand_rows.append({
                'GRAND_ID': filtered.iloc[i].GRAND_ID,
                'MONTH_INDEX': m,
                'MEAN_DEMAND': demand_allocations.iloc[i]
            })
    # write to file
    demand = pd.DataFrame(demand_rows).sort_values(['GRAND_ID', 'MONTH_INDEX'])
    demand.to_parquet(average_monthly_demand_output_path)

    # find the monthly average flow at dam locations
    # TODO subtract sum of upstream demand across each reservoir ??
    flow = xr.open_mfdataset(f"{flow_path}/*.nc")[[flow_key]].load()
    mean_monthly_flow = flow[flow_key].groupby(f'{flow_time_key}.month').mean().compute()
    flow.close()
    columns = mean_monthly_flow.shape[-1]
    flow_rows = []
    for i in np.arange(filtered.index.size):
        flat_index = filtered.iloc[i].GRID_CELL_INDEX
        for m, mean_flow in enumerate(mean_monthly_flow.values[:, flat_index // columns, flat_index % columns]):
            flow_rows.append({
                'GRAND_ID': filtered.iloc[i].GRAND_ID,
                'MONTH_INDEX': m,
                'MEAN_FLOW': mean_flow,
            })
    # write to file
    flow = pd.DataFrame(flow_rows).sort_values(['GRAND_ID', 'MONTH_INDEX'])
    flow.to_parquet(average_monthly_flow_output_path)

    # cast boolean string fields to boolean
    # rely on the fact that None casts to False but any string casts to True
    filtered = filtered.astype({
        'USE_IRRI': bool,
        'USE_ELEC': bool,
        'USE_SUPP': bool,
        'USE_FCON': bool,
        'USE_RECR': bool,
        'USE_NAVI': bool,
        'USE_FISH': bool,
        'USE_PCON': bool,
        'USE_OTHR': bool,
    })

    # join with the istarf parameters
    filtered = filtered.merge(
        istarf[['GRanD_ID'] + list(istarf_key_map.keys())].rename(columns=istarf_key_map),
        how='left',
        left_on='GRAND_ID',
        right_on='GRanD_ID',
    )

    # write reservoir parameters to file
    filtered[
        ['GRAND_ID', 'GRID_CELL_INDEX', 'RES_NAME', 'DAM_NAME', 'RIVER', 'YEAR', 'DAM_HGT_M', 'DAM_LEN_M', 'AREA_SKM',
         'CAP_MCM', 'DEPTH_M', 'CATCH_SKM', 'USE_IRRI', 'USE_ELEC', 'USE_SUPP', 'USE_FCON', 'USE_RECR', 'USE_NAVI',
         'USE_FISH', 'USE_PCON', 'USE_OTHR', 'MAIN_USE', 'LONG_DD',
         'LAT_DD'] + list(istarf_key_map.values())].to_xarray().to_netcdf(reservoir_output_path)

    # write dependency database to file
    dependency_database.to_parquet(dependency_output_path)

    return filtered, grand, grid, flow, demand


def compare_reservoir_placement(
    grid,
    old_parameter_path,
    new_parameters,
    istarf_path,
    grand_id,
):
    old_parameter = xr.open_dataset(old_parameter_path).DamID_Spatial.to_dataframe().reset_index()
    istarf = pd.read_csv(istarf_path)
    kdtree = KDTree(np.array(list(grid.geometry.apply(lambda p: (p.x, p.y)))))
    dam = new_parameters[new_parameters.GRAND_ID == grand_id].set_crs(epsg=4326).to_crs(epsg=3857)
    new_grid_index = kdtree.query((dam.geometry.values[0].x, dam.geometry.values[0].y), k=1)[1]
    old_dam_id = istarf[istarf.GRanD_ID == grand_id].DamID_Spatial.values[0]
    old_grid_index = old_parameter[old_parameter.DamID_Spatial == old_dam_id].index.values[0]
    nearby_cell_indices = kdtree.query((dam.geometry.values[0].x, dam.geometry.values[0].y), k=25)[1]
    cells = []
    offset = 0.125 / 2
    for i in nearby_cell_indices:
        centroid = grid[grid.GRID_CELL_INDEX == i].to_crs(epsg=4326).geometry.values[0]
        cells.append(Polygon([(centroid.x - offset, centroid.y - offset), (centroid.x - offset, centroid.y + offset),
                              (centroid.x + offset, centroid.y + offset), (centroid.x + offset, centroid.y - offset)]))
    cells = gpd.GeoDataFrame(data=grid.iloc[nearby_cell_indices], geometry=cells)
    fig, ax = plt.subplots(figsize=(20, 20))
    dam.set_geometry('geometry').to_crs(epsg=3857).plot(ax=ax, color='blue', alpha=0.125)
    dam.set_geometry('geometry').to_crs(epsg=3857).boundary.plot(ax=ax, color='blue')
    dam.set_geometry('DAM_POINT').to_crs(epsg=3857).plot(color='blue', ax=ax, marker='*')
    dam.set_geometry('RESERVOIR_CENTROID').set_crs(epsg=4326, allow_override=True)\
        .to_crs(epsg=3857).plot(color='purple', ax=ax, marker='d')
    dam.set_geometry('REPRESENTATIVE_POINT').set_crs(epsg=4326, allow_override=True)\
        .to_crs(epsg=3857).plot(color='purple', ax=ax, marker='D')
    grid[grid.GRID_CELL_INDEX == new_grid_index].plot(color='green', ax=ax, marker='o')
    grid[grid.GRID_CELL_INDEX == old_grid_index].plot(color='red', ax=ax, marker='x')
    cells.set_crs(epsg=4326).to_crs(epsg=3857).boundary.plot(ax=ax, color='black')
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager)
    plt.show()
