import click
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
from scipy.spatial import KDTree


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
@click.option(
    '--placement-corrections-path',
    default=None,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    prompt='What is the path to a CSV file containing reservoir placement corrections? Leave blank for none'
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
        placement_corrections_path=None,
        upscale_elevation=False,
        grid_longitude_key='lon',
        grid_latitude_key='lat',
        grid_downstream_key='dnID',
        istarf_grand_id_key='GRanD_ID',
        istarf_observed_meanflow_key='Obs_MEANFLOW_CUMECS',
        demand_key='totalDemand',
        demand_time_key='time',
        flow_key='channel_inflow',
        flow_time_key='time',
        grand_drainage_area_key='CATCH_SKM',
        grid_drainage_area_key='areaTotal',
        elevation_key='hydroshed_average_elevation',
        elevation_upscale_cells=225,
        dependency_radius_meters=200000,
        corrections_grid_index_key='gindex',
        corrections_grand_id_key='GRAND_ID',
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
    grid['DOWNSTREAM_INDEX'] = domain[grid_downstream_key].values.flatten().astype(np.int64) - 1
    grid['DRAINAGE_AREA'] = domain[grid_drainage_area_key].values.flatten() / 1000000

    # create a dam geometry point column
    grand['DAM_POINT'] = grand.apply(lambda dam: Point(dam.LONG_DD, dam.LAT_DD), axis=1)
    grand['RESERVOIR_CENTROID'] = grand.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
    grand['REPRESENTATIVE_POINT'] = grand.to_crs(epsg=3857).representative_point().to_crs(epsg=4326)

    domain.close()

    # remove grand dams that do not appear in the ISTARF database
    grand = grand[grand.GRAND_ID.isin(istarf[istarf_grand_id_key])].set_geometry(
        'DAM_POINT').copy().reset_index(drop=True)
    click.echo(f'GRAND dams appearing in ISTARF data: {len(grand)}')

    # project the grid to web mercator
    grid = grid.set_crs(epsg=4326).to_crs(epsg=3857)

    # create the kdtree to search nearest neighbors
    kdtree = KDTree(np.array(list(grid.geometry.apply(lambda p: (p.x, p.y)))))

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

    # find nearest domain index for each GRanD dam
    grand['GRID_CELL_INDEX'] = grand.set_crs(epsg=4326).to_crs(epsg=3857).geometry.apply(
        lambda p: kdtree.query((p.x, p.y), k=1)[1]
    )
    grand['ORIGINAL_GRID_CELL_INDEX'] = grand.GRID_CELL_INDEX.values.copy()

    # calculate meanflow using output from a non regulated simulation
    # TODO subtract demand from flow ??
    flow = xr.open_mfdataset(f"{flow_path}/*.nc")[[flow_key]].load()
    mean_monthly_flow = flow[flow_key].groupby(f'{flow_time_key}.month').mean()
    meanflow = flow[flow_key].mean(dim=flow_time_key).values.flatten()
    flow.close()

    info = dict(
        has_observed_meanflow=[],
        observed_meanflow_high_error=[],
        observed_meanflow_moves=[],
        drainage_high_error=[],
        drainage_moves=[],
    )

    def move_dam(dam, error_threshold=1.0, improvement_threshold=0.5):
        # if ISTARF has observed mean flow for this reservoir, see how well it matches this grid cell
        # otherwise compare drainage area
        # if error is high, and moving dam would reduce error significantly, then move it
        # don't allow moves to 0 meanflow cells
        # try to force a move if meanflow is already zero
        istarf_observed_meanflow = istarf[istarf_observed_meanflow_key][
            istarf[istarf_grand_id_key] == dam.GRAND_ID].values[0]
        has_no_meanflow = (meanflow[dam.GRID_CELL_INDEX] == 0)
        if np.isfinite(istarf_observed_meanflow):
            info['has_observed_meanflow'].append(dam.GRAND_ID)
            error = abs(istarf_observed_meanflow - meanflow[dam.GRID_CELL_INDEX]) / istarf_observed_meanflow
            if (error >= error_threshold) or has_no_meanflow:
                info['observed_meanflow_high_error'].append(dam.GRAND_ID)
                _, nearest_cells = kdtree.query((dam.DAM_POINT.x, dam.DAM_POINT.y), k=9)
                nearest_cells = nearest_cells[meanflow[nearest_cells] > 0]
                if len(nearest_cells) > 0:
                    _, flow_match = KDTree(meanflow[nearest_cells][:, None]).query(istarf_observed_meanflow, k=1)
                    if nearest_cells[flow_match] != dam.GRID_CELL_INDEX:
                        new_error = abs(istarf_observed_meanflow - meanflow[
                            nearest_cells[flow_match]]) / istarf_observed_meanflow
                        if (new_error / error) <= improvement_threshold:
                            info['observed_meanflow_moves'].append(dam.GRAND_ID)
                            return nearest_cells[flow_match]
        else:
            grand_drainage_area = getattr(dam, grand_drainage_area_key)
            error = abs(grand_drainage_area - grid.iloc[dam.GRID_CELL_INDEX].DRAINAGE_AREA) / grand_drainage_area
            if (error >= error_threshold) or has_no_meanflow:
                info['drainage_high_error'].append(dam.GRAND_ID)
                _, nearest_cells = kdtree.query((dam.DAM_POINT.x, dam.DAM_POINT.y), k=9)
                nearest_cells = nearest_cells[meanflow[nearest_cells] > 0]
                if len(nearest_cells) > 0:
                    _, drainage_area_match = KDTree(grid.iloc[
                        nearest_cells].DRAINAGE_AREA.values.flatten()[:, None]).query(grand_drainage_area, k=1)
                    if nearest_cells[drainage_area_match] != dam.GRID_CELL_INDEX:
                        new_error = abs(grand_drainage_area - grid.iloc[
                            nearest_cells[drainage_area_match]].DRAINAGE_AREA) / grand_drainage_area
                        if (new_error / error) <= improvement_threshold:
                            info['drainage_moves'].append(dam.GRAND_ID)
                            return nearest_cells[drainage_area_match]
        return dam.GRID_CELL_INDEX

    grand['GRID_CELL_INDEX'] = grand.set_crs(epsg=4326).to_crs(epsg=3857).apply(lambda dam: move_dam(dam), axis=1)

    click.echo(f'GRanD dams with observed flow: {len(info["has_observed_meanflow"])}')
    click.echo(f'GRanD dams with high placement error of observed flow: {len(info["observed_meanflow_high_error"])}')
    click.echo(f' - GRanD dams moved based on observed flow: '
               f'{len(info["observed_meanflow_moves"])} - {info["observed_meanflow_moves"]}')
    click.echo(f'GRanD dams with high placement error of drainage area: {len(info["drainage_high_error"])}')
    click.echo(f' - GRanD dams moved based on drainage area: {len(info["drainage_moves"])} - {info["drainage_moves"]}')

    info['has_observed_meanflow'] = []
    info['observed_meanflow_high_error'] = []
    info['observed_meanflow_moves'] = []
    info['drainage_high_error'] = []
    info['drainage_moves'] = []

    # for dams appearing in the same grid cell, try moving again with lower thresholds
    for _, group in grand.set_crs(epsg=4326).to_crs(epsg=3857).reset_index().groupby('GRID_CELL_INDEX', as_index=False):
        if len(group) > 1:
            for k in np.arange(len(group)):
                move_to = move_dam(group.iloc[k], 0.5, 1.0)
                if move_to != group.iloc[k].GRID_CELL_INDEX:
                    grand.at[group.iloc[k]['index'], 'GRID_CELL_INDEX'] = move_to

    click.echo(f'Overlapping GRanD dams with observed flow: {len(info["has_observed_meanflow"])}')
    click.echo(f'Overlapping GRanD dams with high placement error of observed flow: '
               f'{len(info["observed_meanflow_high_error"])}')
    click.echo(f' - GRanD dams moved based on observed flow: '
               f'{len(info["observed_meanflow_moves"])} - {info["observed_meanflow_moves"]}')
    click.echo(f'Overlapping GRanD dams with high plaecment error of drainage area: {len(info["drainage_high_error"])}')
    click.echo(f' - GRanD dams moved based on drainage area: {len(info["drainage_moves"])} - {info["drainage_moves"]}')

    # for dams still appearing in the same grid cell,
    # move overlaps up or downstream based on drainage area,
    # but still only allow moves to cells that have > 0 meanflow
    # iterate thrice
    moved_upstream = []
    moved_downstream = []
    for _ in np.arange(3):
        grand = grand.sort_values(['CAP_MCM', 'AREA_SKM', 'GRAND_ID'], ascending=False).reset_index(drop=True)
        for _, group in grand.reset_index().groupby('GRID_CELL_INDEX', as_index=False):
            if len(group) > 1:
                for k in np.arange(len(group))[1:]:
                    if getattr(group.iloc[k], grand_drainage_area_key) > getattr(group.iloc[0], grand_drainage_area_key):
                        new_index = grid.iloc[group.iloc[k].GRID_CELL_INDEX].DOWNSTREAM_INDEX
                        # if new index is less than 0, dam would be moved into the ocean... so don't allow this.
                        if new_index >= 0:
                            # move downstream
                            moved_downstream.append(group.iloc[k].GRAND_ID)
                            grand.at[group.iloc[k]['index'], 'GRID_CELL_INDEX'] = new_index
                    else:
                        # move upstream
                        # there can be multiple upstream cells, so move to the best flow or drainage area match
                        upstream_cells = grid[grid.DOWNSTREAM_INDEX == group.iloc[k].GRID_CELL_INDEX]
                        istarf_observed_meanflow = istarf[istarf_observed_meanflow_key][
                            istarf[istarf_grand_id_key] == group.iloc[k].GRAND_ID].values[0]
                        upstream_meanflow = meanflow[upstream_cells.GRID_CELL_INDEX.values]
                        if np.isfinite(istarf_observed_meanflow):
                            match = -1
                            min_error = np.Inf
                            for i, f in enumerate(upstream_meanflow):
                                if f == 0:
                                    continue
                                e = abs(f - istarf_observed_meanflow)
                                if e < min_error:
                                    match = upstream_cells.iloc[[i]].index[0]
                                    min_error = e
                            if match > -1:
                                moved_upstream.append(group.iloc[k].GRAND_ID)
                                grand.at[group.iloc[k]['index'], 'GRID_CELL_INDEX'] = match
                        else:
                            grand_drainage_area = getattr(group.iloc[k], grand_drainage_area_key)
                            upstream_drainage_area = upstream_cells.DRAINAGE_AREA.values
                            match = -1
                            min_error = np.Inf
                            for i, d in enumerate(upstream_drainage_area):
                                if upstream_meanflow[i] == 0:
                                    continue
                                e = abs(d - grand_drainage_area)
                                if e < min_error:
                                    match = upstream_cells.iloc[[i]].index[0]
                                    min_error = e
                            if match > -1:
                                moved_upstream.append(group.iloc[k].GRAND_ID)
                                grand.at[group.iloc[k]['index'], 'GRID_CELL_INDEX'] = match

    click.echo(f'Overlapping GRanD dams moved upstream: {len(moved_upstream)} - {moved_upstream}')
    click.echo(f'Overlapping GRanD dams moved downstream: {len(moved_downstream)} - {moved_downstream}')

    # finally, remove remaining duplicate dams, preferring largest capacity, then largest drainage area, then latest id
    filtered = grand.groupby('GRID_CELL_INDEX', as_index=False, group_keys=False).first()
    dropped = grand.groupby('GRID_CELL_INDEX', as_index=False, group_keys=False).nth(np.arange(20).tolist()[1:])

    click.echo(f'GRanD dams dropped due to overlaps: {len(dropped.index)} - {dropped.GRAND_ID.values.tolist()}')
    click.echo(f'GRanD dams remaining after removing overlaps: {len(filtered.index)}')

    # if provided, implement placement corrections
    # note that this currently doesn't correct any manual placement errors (duplicates will just get dropped)
    if (placement_corrections_path is not None) and (placement_corrections_path != ''):
        click.echo(f'Applying corrections to GRanD parameters.')
        corrections = pd.read_csv(placement_corrections_path)[[corrections_grand_id_key, corrections_grid_index_key]]
        for i in np.arange(len(corrections)):
            # if another dam is in this same grid cell, remove it
            filtered = filtered.drop(filtered[filtered['GRID_CELL_INDEX'] == corrections.iloc[i][corrections_grid_index_key]].index)
            if corrections.iloc[i][corrections_grand_id_key] in filtered['GRAND_ID'].values:
                filtered.loc[
                    filtered['GRAND_ID'] == corrections.iloc[i][corrections_grand_id_key],
                    'GRID_CELL_INDEX'
                ] = int(corrections.iloc[i][corrections_grid_index_key])
            else:
                # get the rest of the parameters from the grand file
                filtered = pd.concat(
                    [filtered, grand[grand['GRAND_ID'] == corrections.iloc[i][corrections_grand_id_key]][filtered.columns]],
                    ignore_index=True,
                )
                filtered.loc[
                    filtered['GRAND_ID'] == corrections.iloc[i][corrections_grand_id_key],
                    'GRID_CELL_INDEX'
                ] = int(corrections.iloc[i][corrections_grid_index_key])

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
        dependent_cells = dependent_cells[dependent_cells['ELEVATION'] <= dam_cell['ELEVATION']]
        dependent_cells = dependent_cells[dependent_cells['OUTLET_INDEX'] == dam_cell['OUTLET_INDEX']]
        dependent_cells = dependent_cells.index.values
        dependent_cell_indices.append(dependent_cells[~np.isnan(dependent_cells)].astype(np.int64))
    filtered['DEPENDENT_CELL_INDICES'] = dependent_cell_indices

    filtered.index = filtered.GRAND_ID.values
    filtered = filtered.sort_index()

    dependency_database = filtered[['GRAND_ID', 'DEPENDENT_CELL_INDICES']].explode(
        column='DEPENDENT_CELL_INDICES').rename(
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
    flow = pd.DataFrame(flow_rows).sort_values(['GRAND_ID', 'MONTH_INDEX'])

    # log dams with no annual average meanflow
    no_flow = flow.groupby('GRAND_ID').mean()
    no_flow = no_flow[no_flow.MEAN_FLOW == 0].index.tolist()
    click.echo(f'GRanD dams with no mean inflow: {len(no_flow)} - {no_flow}')

    # write to file
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
         'USE_FISH', 'USE_PCON', 'USE_OTHR', 'MAIN_USE', 'ORIGINAL_GRID_CELL_INDEX', 'LONG_DD',
         'LAT_DD'] + list(istarf_key_map.values())].to_xarray().to_netcdf(reservoir_output_path)

    # write dependency database to file
    dependency_database.to_parquet(dependency_output_path)
