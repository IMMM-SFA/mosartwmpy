import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Polygon, Point
import xarray as xr
import hvplot.xarray # noqa
from hvplot import show
import warnings

from mosartwmpy import Model
from mosartwmpy.utilities.epiweek import get_epiweek_from_datetime

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def plot_variable(
    model: Model,
    variable: str,
    start: str = None,
    end: str = None,
    log_scale: bool = False,
    cmap: str = 'autumn_r',
    tiles: str = 'StamenWatercolor',
    width: int = 1040,
    height: int = 720,
    alpha: float = 0.75,
    no_tiles: bool = False,
):
    variable_config = next((o for o in model.config.get('simulation.output') if
                            o.get('name', '').casefold() == variable.casefold()), None)
    if not variable_config:
        variable_config = next((o for o in model.config.get('simulation.output') if
                                o.get('variable', '').casefold() == variable.casefold()), None)
    if not variable_config:
        variable_config = next((o for o in model.config.get('simulation.output') if
                                o.get('long_name', '').casefold() == variable.casefold()), None)
    if not variable_config:
        raise NameError(f'Variable {variable} not found in model configuration.')

    has_cartopy = False
    if not no_tiles:
        try:
            import cartopy
            import scipy
            import geoviews
            has_cartopy = True
        except ModuleNotFoundError:
            # Error handling
            pass

    results = xr.open_mfdataset(
        f"{model.config.get('simulation.output_path')}/{model.name}/*.nc")[[variable_config.get('name')]].sel(
            time=slice(start, end)).load()

    vmin = (1 if log_scale else 0)
    vmax = results[variable_config.get('name')].max().values[()]

    plot = results.where(results > vmin)[variable_config.get('name')].hvplot.quadmesh(
        'lon',
        'lat',
        variable_config.get('name'),
        coastline=has_cartopy,
        geo=has_cartopy,
        tiles=tiles if has_cartopy else False,
        width=width,
        height=height,
        cmap=cmap,
        alpha=alpha,
        clim=(vmin, vmax),
        cnorm='log' if log_scale else 'linear',
        title=f"{variable_config.get('long_name')} ({variable_config.get('units')})",
    )

    show(plot)


def plot_reservoir(
    model: Model,
    grand_id: int,
    grand_file_path: str = None,
    istarf_data_path: str = None,
    start: str = None,
    end: str = None,
):

    spacing = model.get_grid_spacing()
    offset_x = spacing[0] / 2
    offset_y = spacing[1] / 2

    dam_index = model.unmask(model.grid.reservoir_id).tolist().index(grand_id)
    ilat = dam_index // model.get_grid_shape()[1]
    ilon = dam_index % model.get_grid_shape()[1]
    dependent_cell_indices = model.grid.reservoir_dependency_database.query(
        f'reservoir_id == {grand_id}').index.values
    basin_cell_indices = np.nonzero(model.unmask(model.grid.outlet_id) == model.unmask(model.grid.outlet_id)[dam_index])[0]

    if grand_file_path is not None:
        grand = gpd.read_file(grand_file_path)
        grand_dam = grand[grand.GRAND_ID == grand_id].iloc[0]
        grand_point = Point(grand_dam.LONG_DD, grand_dam.LAT_DD)
        grand_reservoir = grand_dam.geometry
    else:
        grand_point = None
        grand_reservoir = None

    stream_indices = [dam_index]
    j = dam_index
    while True:
        i = model.unmask(model.grid.downstream_id)[j]
        if (i >= 0) and (i != j):
            stream_indices.append(i)
            j = i
        else:
            break
    stream_indices = np.array(stream_indices)

    to_plot = [{
        'geometry': Point(model.unmask(model.grid.longitude)[dam_index], model.unmask(model.grid.latitude)[dam_index]),
        'type': 'dam',
    }]
    if grand_point is not None:
        to_plot.append({
            'geometry': grand_point,
            'type': 'grand_dam',
        })
    if grand_reservoir is not None:
        to_plot.append({
            'geometry': grand_reservoir,
            'type': 'grand_reservoir',
        })
    for t, array in dict(dependency=dependent_cell_indices, basin=basin_cell_indices, stream=stream_indices).items():
        for i in array:
            x = model.unmask(model.grid.longitude)[i]
            y = model.unmask(model.grid.latitude)[i]
            to_plot.append({
                'geometry': Polygon([
                    (x - offset_x, y - offset_y),
                    (x - offset_x, y + offset_y),
                    (x + offset_x, y + offset_y),
                    (x + offset_x, y - offset_y),
                ]),
                'type': t,
            })

    to_plot = gpd.GeoDataFrame(to_plot).set_crs(epsg=4326).to_crs(epsg=3857)

    fig = plt.figure()
    grid = fig.add_gridspec(3, 2)
    ax_storage = fig.add_subplot(grid[0, 0])
    ax_inflow = fig.add_subplot(grid[1, 0])
    ax_outflow = fig.add_subplot(grid[2, 0])
    ax_map = fig.add_subplot(grid[:, 1])

    to_plot[to_plot['type'] == 'basin'].boundary.plot(ax=ax_map, color='#336699', alpha=0.0625)
    to_plot[to_plot['type'] == 'dependency'].plot(ax=ax_map, color='green', alpha=0.125)
    to_plot[to_plot['type'] == 'stream'].plot(ax=ax_map, color='#336699', alpha=0.33)
    if grand_reservoir is not None:
        to_plot[to_plot['type'] == 'grand_reservoir'].plot(ax=ax_map, color='#336699', alpha=0.5)
    if grand_point is not None:
        to_plot[to_plot['type'] == 'grand_dam'].plot(ax=ax_map, color='#9F2B68', marker='*')
    to_plot[to_plot['type'] == 'dam'].plot(ax=ax_map, color='#FA8072')

    ctx.add_basemap(ax_map, source=ctx.providers.CartoDB.Voyager)

    storage_variable = next((o for o in model.config.get('simulation.output') if
                             o.get('variable', '') == 'reservoir_storage'), {}).get('name', None)
    inflow_variable = next((o for o in model.config.get('simulation.output') if
                            o.get('variable', '') == 'channel_inflow_upstream'), {}).get('name', None)
    outflow_variable = next((o for o in model.config.get('simulation.output') if
                             o.get('variable', '') == 'runoff_land'), {}).get('name', None)
    results = xr.open_mfdataset(f"{model.config.get('simulation.output_path')}/{model.name}/*.nc")[
        [storage_variable, inflow_variable, outflow_variable]]

    df = pd.DataFrame(index=results.time)
    df['storage'] = results[storage_variable].isel(lat=ilat, lon=ilon).values.flatten() / 1e6
    df['inflow'] = results[inflow_variable].isel(lat=ilat, lon=ilon).values.flatten() * 24 * 60 * 60 / 1e6
    df['outflow'] = results[outflow_variable].isel(lat=ilat, lon=ilon).values.flatten() * 24 * 60 * 60 / 1e6
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    results.close()

    params = xr.open_dataset(model.config.get('water_management.reservoirs.parameters.path')).to_dataframe()
    dam = params[params[model.config.get('water_management.reservoirs.parameters.variables.reservoir_id')] == grand_id]
    weeks = df.index.to_series().apply(lambda time: get_epiweek_from_datetime(pd.to_datetime(time)))
    weeks[weeks > 52] = 52

    meanflow = pd.read_parquet(model.config.get('water_management.reservoirs.streamflow.path'))
    mean_weekly_volume = 7.0 * meanflow[meanflow.GRAND_ID == grand_id].MEAN_FLOW.mean() * 24.0 * 60.0 * 60.0
    release_max = mean_weekly_volume * (1 + dam.iloc[0].release_max) / 7.0 / 1e6
    release_min = mean_weekly_volume * (1 + dam.iloc[0].release_min) / 7.0 / 1e6
    ax_outflow.fill_between(df.index, release_min, release_max, alpha=0.25, facecolor='#B2BEB5')
    max_storage = np.minimum(
        dam.iloc[0].upper_max,
        np.maximum(
            dam.iloc[0].upper_min,
            dam.iloc[0].upper_mu +
            dam.iloc[0].upper_alpha * np.sin(2.0 * np.pi * (1 / 52) * weeks) +
            dam.iloc[0].upper_beta * np.cos(2.0 * np.pi * (1 / 52) * weeks)
        )
    ) * dam.iloc[0].CAP_MCM / 100
    min_storage = np.minimum(
        dam.iloc[0].lower_max,
        np.maximum(
            dam.iloc[0].lower_min,
            dam.iloc[0].lower_mu +
            dam.iloc[0].lower_alpha * np.sin(2.0 * np.pi * (1 / 52) * weeks) +
            dam.iloc[0].lower_beta * np.cos(2.0 * np.pi * (1 / 52) * weeks)
        )
    ) * dam.iloc[0].CAP_MCM / 100
    ax_storage.fill_between(df.index, min_storage, max_storage, alpha=0.25, facecolor='#B2BEB5')

    if istarf_data_path is not None:
        istarf = pd.read_csv(istarf_data_path)
        istarf['date'] = pd.to_datetime(istarf.date, format='%Y-%m-%d')
        istarf = istarf[istarf.GRAND_ID == grand_id]
        istarf = istarf[(istarf.date >= df.index.min()) & (istarf.date <= df.index.max())]
        istarf = istarf.sort_values('date').set_index('date')
        istarf[['s']].rename(columns={'s': 'observed'}).plot(
            ax=ax_storage,
            xlabel='date',
            ylabel='MCM',
            kind='line',
            title=f'{dam.DAM_NAME.iloc[0]} ({grand_id}) - Storage',
            color='#D22B2B',
            linestyle='dotted',
            linewidth=1,
            sharex=True,
            alpha=1,
        )
        istarf[['s_free']].rename(columns={'s_free': 'istarf'}).plot(
            ax=ax_storage,
            xlabel='date',
            ylabel='MCM',
            kind='line',
            title=f'{dam.DAM_NAME.iloc[0]} ({grand_id}) - Storage',
            color='#4A0404',
            linestyle='dashed',
            linewidth=1,
            sharex=True,
            alpha=1,
        )
        istarf[['i']].rename(columns={'i': 'observed'}).plot(
            ax=ax_inflow,
            xlabel='date',
            ylabel='MCM',
            kind='line',
            title=f'{dam.DAM_NAME.iloc[0]} ({grand_id}) - Storage',
            color='#D22B2B',
            linestyle='dotted',
            linewidth=1,
            sharex=True,
            alpha=1,
        )
        istarf[['r_free']].rename(columns={'r_free': 'istarf'}).plot(
            ax=ax_outflow,
            xlabel='date',
            ylabel='MCM',
            kind='line',
            title=f'{dam.DAM_NAME.iloc[0]} ({grand_id}) - Storage',
            color='#4A0404',
            linestyle='dashed',
            linewidth=1,
            sharex=True,
            alpha=1,
        )
        istarf['outflow'] = istarf['i'] - istarf['s'].diff()
        istarf[['outflow']].rename(columns={'outflow': 'observed'}).plot(
            ax=ax_outflow,
            xlabel='date',
            ylabel='MCM',
            kind='line',
            title=f'{dam.DAM_NAME.iloc[0]} ({grand_id}) - Storage',
            color='#D22B2B',
            linestyle='dotted',
            linewidth=1,
            sharex=True,
            alpha=1,
        )


    df[['storage']].plot(
        ax=ax_storage,
        xlabel='date',
        ylabel='MCM',
        kind='line',
        title=f'{dam.DAM_NAME.iloc[0]} ({grand_id}) - Storage',
        color='#336699',
        linestyle='solid',
        linewidth=1,
        sharex=True,
        alpha=1,
    )
    df[['inflow']].plot(
        ax=ax_inflow,
        xlabel='date',
        ylabel='MCM',
        kind='line',
        title=f'{dam.DAM_NAME.iloc[0]} ({grand_id}) - Inflow',
        color='#336699',
        linestyle='solid',
        linewidth=1,
        sharex=True,
        alpha=1,
    )
    df[['outflow']].plot(
        ax=ax_outflow,
        xlabel='date',
        ylabel='MCM',
        kind='line',
        title=f'{dam.DAM_NAME.iloc[0]} ({grand_id}) - Outflow',
        color='#336699',
        linestyle='solid',
        linewidth=1,
        sharex=True,
        alpha=1,
    )

    fig.tight_layout()
    plt.show()






