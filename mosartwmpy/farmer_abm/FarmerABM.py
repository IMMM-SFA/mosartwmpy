from benedict.dicts import benedict
import netCDF4
import logging
import math
import numpy as np
from os.path import exists
import pandas as pd
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pyutilib.subprocess.GlobalData
import shutil
import sys
import xarray as xr

from mosartwmpy.config.config import get_config

import pdb


class FarmerABM:

    def __init__(self, config: benedict, config_path: str = None):
        if config_path is not None:
            self.config = get_config(config_path)
        else:
            self.config = config 
        self.mu = config.get('water_management.demand.farmer_abm.mu', 0.2)
        self.processed_years = []


    def calc_demand(self, name, year):
        month = '1'
        reservoir_file_path =  './legacy_reservoir_file.nc'
        code_path = self.config.get('water_management.demand.farmer_abm.path')

        logging.info('code path: ' + code_path)

        pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False
        output_dir = f"{self.config.get('simulation.output_path')}/demand/"
        output_path = f"{output_dir}{name}_farmer_abm_demand_{year}_{month}.nc"
        months = ['01','02','03','04','05','06','07','08','09','10','11','12']

        # Check that we haven't already performed the farmer ABM calculation.
        if year in self.processed_years:
            logging.debug('Already performed farmer ABM calculations for ' + year + '. Files are in ' + output_dir)
            return

        logging.debug('sys version: ' + str(sys.version_info))
        logging.debug('pandas version: ' + pd.__version__)
        try:
            logging.info(f'Using reservoir parameter file: {reservoir_file_path}')

            with open(code_path + '/nldas_ids.p', 'rb') as fp:
                nldas_ids = pickle.load(fp)

            nldas = pd.read_csv(code_path+ '/nldas.txt')

            with open(code_path+'/water_constraints_by_farm_pyt278.p', 'rb') as fp:
                water_constraints_by_farm = pickle.load(fp)
            # water_constraints_by_farm = pd.read_pickle('/pic/projects/im3/wm/Jim/pmp_input_files/water_constraints_by_farm_v2.p')
            water_constraints_by_farm = dict.fromkeys(water_constraints_by_farm, 9999999999)

            ## Read in Water Availability Files from MOSART-PMP
            if year < 1950:  # If year is before 1950 (warm-up period), use the external baseline water demand files
                water_constraints_by_farm = pd.read_csv(code_path+'/hist_avail_bias_correction_20201102.csv') # Use baseline water demand data for warmup period
                water_constraints_by_farm = water_constraints_by_farm[['NLDAS_ID','sw_irrigation_vol']].reset_index()
                water_constraints_by_farm = water_constraints_by_farm['sw_irrigation_vol'].to_dict()
            # elif year>=1950:  # For first year of ABM, use baseline water demand data
            elif year == 1950:
                water_constraints_by_farm = pd.read_csv(code_path+'/hist_avail_bias_correction_20201102.csv')
                water_constraints_by_farm = water_constraints_by_farm[['NLDAS_ID','sw_irrigation_vol']].reset_index()
                water_constraints_by_farm = water_constraints_by_farm['sw_irrigation_vol'].to_dict()
            else:
                # loop through .nc files and extract data
                # first = True
                # for m in months:
                #     #dataset_name = 'jim_abm_integration.mosart.h0.' + str(year-1) + '-' + m + '.nc'
                #     logging.info(f'Trying to load WM output for month {month} year {year}')
                #     # dataset_name = case_name +'.mosart.h0.' + str(year - 1) + '-' + m + '.nc'
                #     dataset_name = name +'.mosart.h0.' + str(year - 1) + '-' + m + '.nc'
                #     logging.info(f'Successfully load WM output for month {month} year {year}')
                #     # ds = xr.open_dataset(output_path+'/'+dataset_name)
                #     ds = xr.open_dataset('/Users/rexe871/Downloads/mosart_output_files_1982/istarf_validation_1982_01.nc')
                #     logging.info(f'Successfully converted to df for month {month} year {year}')
                #     df = df.reset_index()
                #     df_merge = pd.merge(df, nldas, how='left', left_on=['lat', 'lon'], right_on=['CENTERY', 'CENTERX'])
                #     logging.info(f'Successfully abm_supply_availd df for month {month} year {year}')

                #     df_select = df_merge[['abm_supply_availD', 'WRM_DEMAND0', 'WRM_SUPPLY', 'WRM_DEFICIT','WRM_STORAGE','GINDEX','RIVER_DISCHARGE_OVER_LAND_LIQ']]
                #     logging.info(f'Successfully subsetted df for month {month} year {year}')
                #     df_select['year'] = year
                #     df_select['month'] = int(m)
                #     if first:
                #         df_all = df_select
                #         first = False
                #     else:
                #         df_all = pd.concat([df_all, df_select])

                # # calculate average across timesteps
                # # df_pivot = pd.pivot_table(df_all, index=['NLDAS_ID','GINDEX'], values=['WRM_SUPPLY','WRM_STORAGE','RIVER_DISCHARGE_OVER_LAND_LIQ'],
                # #                           aggfunc=np.abm_supply_avail)  # units will be average monthly (m3/s)

                # df_pivot = df_pivot.reset_indexabm_supply_avail)


                # df_pivot = df_pivot[df_pivotabm_supply_avail'NLDAS_ID'].isin(nldas_ids)].reset_index()
                # df_pivot.fillna(0)
                # logging.info(f'Successfully pivoted df for month {month} year {year}')

                # # calculate dependent storage
                # ds = xr.open_dataset(reservoir_file_path)
                # dams = ds["DamInd_2d"].to_dataframe()
                # dams = dams.reset_index()
                # dep = ds["gridID_from_Dam"].to_dataframe()
                # dep_merge = pd.merge(dep, dams, how='left', left_on=['Dams'], right_on=['DamInd_2d'])
                
                # df_pivot = pd.merge(df_pivot, nldas, how='left', on='NLDAS_ID')

                # dep_merge = pd.merge(abm_supply_availe, df_pivot[['NLDAS_ID','CENTERX','CENTERY','WRM_STORAGE','RIVER_DISCHARGE_OVER_LAND_LIQ']], how='left', left_on=['lat','lon'], right_on=['CENTERY','CENTERX'])
                # dep_merge['WRM_STORAGE'] = dep_merge['WRM_STORAGE'].fillna(0)

                # aggregation_functions = {'WRM_STORAGE': 'sum'}
                # dep_merge = dep_merge.groupby(['gridID_from_Dam'], as_index=False).aggregate(aggregation_functions)
                # wm_results = pd.merge(df_pivot, dep_merge, how='left', left_on=['GINDEX'], right_on=['gridID_from_Dam'])
                
                # abm_supply_avail = wm_results[wm_results['NLDAS_ID'].isin(nldas_ids)].reset_index()

                # abm_supply_avail = abm_supply_avail[['abm_supply_availY','NLDAS_ID','STORAGE_SUM','RIVER_DISCHARGE_OVER_LAND_LIQ']]
                # abm_supply_avail = abm_supply_avail.fillna(0)

                # travis alternative way of getting desired df

                dependency_database_path = 'input/reservoirs/dependency_database.parquet'
                # reservoir_parameter_path = 'input/reservoirs/grand_reservoir_parameters.nc'
                reservoir_parameter_path = 'input/reservoirs/reservoirs.nc'
                simulation_output_path = 'output/istarf_validation/istarf_validation_1982*.nc'

                # relationships between Grid Cells and reservoirs they can consume from (many to many)
                dependency_database = pd.read_parquet(dependency_database_path)

                # determines which grid cells the reservoirs are located in (one to one)
                reservoir_parameters = xr.open_dataset(reservoir_parameter_path)[['GRAND_ID', 'GRID_CELL_INDEX']].to_dataframe()

                # merge the dependencies with the reservoir grid cells
                dependency_database = dependency_database.merge(reservoir_parameters, how='left', on='GRAND_ID').rename(columns={'GRID_CELL_INDEX': 'RESERVOIR_CELL_INDEX'})

                # the output data from the simulation (per grid cell)
                # simulation_output = xr.open_mfdataset(simulation_output_path)[['GINDEX', 'WRM_STORAGE', 'WRM_SUPPLY', 'RIVER_DISCHARGE_OVER_LAND_LIQ']].to_dataframe().reset_index()
                simulation_output = xr.open_mfdataset(simulation_output_path)[['GINDEX', 'WRM_STORAGE', 'WRM_SUPPLY', 'RIVER_DISCHARGE_OVER_LAND_LIQ', 'lat', 'lon']].to_dataframe().reset_index()


                # take the mean of the simulation outputs over the year
                simulation_output = simulation_output.groupby('GINDEX').mean().reset_index()
                
                # erx
                simulation_output.fillna(0)

                # need to join the grid cells with the total reservoir water they had access to
                abm_data = pd.merge(dependency_database, simulation_output[['GINDEX', 'WRM_STORAGE']], how='left', left_on='RESERVOIR_CELL_INDEX', right_on='GINDEX')
                abm_data = abm_data.groupby('DEPENDENT_CELL_INDEX').aggregate({'WRM_STORAGE': 'sum'})

                # now need to join the grid cells with their amount of water they were actually supplied, and the amount of water flowing through the cell
                abm_data = abm_data.rename(columns={'WRM_STORAGE': 'STORAGE_SUM'})
                abm_data = pd.merge(abm_data, simulation_output[['GINDEX', 'WRM_SUPPLY', 'RIVER_DISCHARGE_OVER_LAND_LIQ', 'lat', 'lon']], how='left', left_on='DEPENDENT_CELL_INDEX', right_on='GINDEX')

                abm_data = abm_data.reset_index().rename(columns={'index': 'GRID_CELL_INDEX'})

                # convert lat and lon into nldas ID
                MIN_LON_IN_NLDAS = -124.9375
                MIN_LAT_IN_NLDAS = 25.0625
                NLDAS_GRID_DEFINITION = 8
                abm_data['NLDAS_ID'] = ['x'+str(int((row['lon']-MIN_LON_IN_NLDAS)*NLDAS_GRID_DEFINITION+1))+'y'+str(int((row['lat']-MIN_LAT_IN_NLDAS)*NLDAS_GRID_DEFINITION+1)) for _,row in abm_data.iterrows()]
                abm_data = abm_data.drop(['lat', 'lon'], axis=1)

                print(abm_data)

                # temp var to test if travis' code works
                abm_supply_avail = abm_data

                # print("starting row operations to calculate nldas id")
                # abm_supply_avail['NLDAS_ID'] = ['x'+str(int((row['lon']+124.9375)*8+1))+'y'+str(int((row['lat']-25.0625)*8+1)) for _,row in abm_supply_avail.iterrows()] 


                # print(abm_supply_avail)

                # df_merge = pd.merge(abm_supply_avail, nldas, how='left', left_on=['lat', 'lon'], right_on=['CENTERY', 'CENTERX'])
                # df_pivot = pd.merge(abm_supply_avail, nldas, how='left', on='NLDAS_ID')

                # pdb.set_trace()

                # convert units from m3/s to acre-ft/yr

                if year == '1951':
                    hist_avail_bias = pd.read_csv(code_path+'/hist_avail_bias_correction_20201102.csv')
                    hist_avail_bias['WRM_SUPPLY_acreft_prev'] = hist_avail_bias['WRM_SUPPLY_acreft_OG']
                else:
                    hist_avail_bias = pd.read_csv(code_path+'/hist_avail_bias_correction_live.csv')

                    # erx
                    hist_avail_bias['WRM_SUPPLY_acreft_prev'] = hist_avail_bias['WRM_SUPPLY_acreft_prev'].fillna(0)

                hist_storage = pd.read_csv(code_path+'/hist_dependent_storage.csv')
                hist_avail_bias = pd.merge(hist_avail_bias, hist_storage, how='left', on='NLDAS_ID')

                # print(hist_avail_bias)
                # print(hist_avail_bias.columns)

                abm_supply_avail = pd.merge(abm_supply_avail, hist_avail_bias[['NLDAS_ID','sw_avail_bias_corr','WRM_SUPPLY_acreft_OG','WRM_SUPPLY_acreft_prev','RIVER_DISCHARGE_OVER_LAND_LIQ_OG','STORAGE_SUM_OG']], on=['NLDAS_ID'])
                # abm_supply_avail['demand_factor'] = abm_supply_avail['STORAGE_SUM'] / abm_supply_avail['STORAGE_SUM_OG']
                # abm_supply_avail['demand_factor'] = np.where(abm_supply_avail['STORAGE_SUM_OG'] > 0, abm_supply_avail['STORAGE_SUM'] / abm_supply_avail['STORAGE_SUM_OG'],
                abm_supply_avail = abm_supply_avail.assign(demand_factor=np.where((abm_supply_avail.STORAGE_SUM_OG>0) & (pd.notnull(abm_supply_avail.STORAGE_SUM)), abm_supply_avail.STORAGE_SUM / abm_supply_avail.STORAGE_SUM_OG,
                                                            # np.where(abm_supply_avail['RIVER_DISCHARGE_OVER_LAND_LIQ_OG'] >= 0.1, 
                                                            np.where((abm_supply_avail['RIVER_DISCHARGE_OVER_LAND_LIQ_OG'] >= 0.1) & (pd.notnull(abm_supply_avail['RIVER_DISCHARGE_OVER_LAND_LIQ_OG'])) & pd.notnull(abm_supply_avail['RIVER_DISCHARGE_OVER_LAND_LIQ']),
                                                                    abm_supply_avail['RIVER_DISCHARGE_OVER_LAND_LIQ'] / abm_supply_avail['RIVER_DISCHARGE_OVER_LAND_LIQ_OG'],
                                                                    1))
                )

                abm_supply_avail['WRM_SUPPLY_acreft_newinfo'] = abm_supply_avail['demand_factor'] * abm_supply_avail['WRM_SUPPLY_acreft_OG']

                abm_supply_avail['WRM_SUPPLY_acreft_updated'] = ((1 - self.mu) * abm_supply_avail['WRM_SUPPLY_acreft_prev']) + (self.mu * abm_supply_avail['WRM_SUPPLY_acreft_newinfo'])

                abm_supply_avail['WRM_SUPPLY_acreft_prev'] = abm_supply_avail['WRM_SUPPLY_acreft_updated']
                abm_supply_avail[['NLDAS_ID','WRM_SUPPLY_acreft_OG','WRM_SUPPLY_acreft_prev','sw_avail_bias_corr','demand_factor','RIVER_DISCHARGE_OVER_LAND_LIQ_OG']].to_csv(code_path+'/hist_avail_bias_correction_live.csv')
                abm_supply_avail['WRM_SUPPLY_acreft_bias_corr'] = abm_supply_avail['WRM_SUPPLY_acreft_updated'] + abm_supply_avail['sw_avail_bias_corr']
                water_constraints_by_farm = abm_supply_avail['WRM_SUPPLY_acreft_bias_corr'].to_dict()
                logging.info(f'Successfully converted units df for month {month} year {year}')

                # pdb.set_trace()

            logging.info(f'I have successfully loaded water availability files for month {month} year {year}.')

            ## Read in PMP calibration files
            data_file=pd.ExcelFile(code_path+"/MOSART_WM_PMP_inputs_20201028.xlsx")
            data_profit = data_file.parse("Profit")
            water_nirs=data_profit["nir_corrected"]
            nirs=dict(water_nirs)

            logging.info(f'I have successfully loaded PMP calibration files for month {month} year {year}.')

            ids = range(538350) # total number of crop and nldas ID combinations
            farm_ids = range(53835) # total number of farm agents / nldas IDs
            with open(code_path+'/crop_ids_by_farm.p', 'rb') as fp:
                crop_ids_by_farm = pickle.load(fp)
                crop_ids_by_farm_and_constraint = np.copy(crop_ids_by_farm)
            with open(code_path+'/max_land_constr_20201102_protocol2.p', 'rb') as fp:
                land_constraints_by_farm = pickle.load(fp)

            # Revise to account for removal of "Fodder_Herb category"
            crop_ids_by_farm_new = {}
            for i in crop_ids_by_farm:
                crop_ids_by_farm_new[i] = crop_ids_by_farm[i][0:10]
            crop_ids_by_farm = crop_ids_by_farm_new
            crop_ids_by_farm_and_constraint = crop_ids_by_farm_new

            # Load gammas and alphas
            with open(code_path+'/gammas_new_20201102_protocol2.p', 'rb') as fp:
                gammas = pickle.load(fp)
            with open(code_path+'/net_prices_new_20201102_protocol2.p', 'rb') as fp:
                net_prices = pickle.load(fp)

            # !JY! replace net_prices with zero value for gammas that equal to zero
            for n in range(len(net_prices)):
                if gammas[n] == 0:
                    net_prices[n] = 0

            x_start_values=dict(enumerate([0.0]*3))

            logging.info(f'I have loaded constructed model indices, constraints for month {month} year {year}.')

            # pdb.set_trace()

            ## C.2. 2st stage: Quadratic model included in JWP model simulations
            ## C.2.a. Constructing model inputs:
            ##  (repetition to be safe - deepcopy does not work on PYOMO models)
            fwm_s = ConcreteModel()
            fwm_s.ids = Set(initialize=ids)
            fwm_s.farm_ids = Set(initialize=farm_ids)
            fwm_s.crop_ids_by_farm = Set(fwm_s.farm_ids, initialize=crop_ids_by_farm)
            fwm_s.crop_ids_by_farm_and_constraint = Set(fwm_s.farm_ids, initialize=crop_ids_by_farm_and_constraint)
            fwm_s.net_prices = Param(fwm_s.ids, initialize=net_prices, mutable=True)
            fwm_s.gammas = Param(fwm_s.ids, initialize=gammas, mutable=True)
            fwm_s.land_constraints = Param(fwm_s.farm_ids, initialize=land_constraints_by_farm, mutable=True)
            fwm_s.water_constraints = Param(fwm_s.farm_ids, initialize=water_constraints_by_farm, mutable=True) #JY here need to read calculate new water constraints
            fwm_s.xs = Var(fwm_s.ids, domain=NonNegativeReals, initialize=x_start_values)
            fwm_s.nirs = Param(fwm_s.ids, initialize=nirs, mutable=True)

            ## C.2.b. 2nd stage model: Constructing functions:
            def obj_fun(fwm_s):
                return 0.00001*sum(sum((fwm_s.net_prices[i] * fwm_s.xs[i] - 0.5 * fwm_s.gammas[i] * fwm_s.xs[i] * fwm_s.xs[i]) for i in fwm_s.crop_ids_by_farm[f]) for f in fwm_s.farm_ids)
            fwm_s.obj_f = Objective(rule=obj_fun, sense=maximize)


            def land_constraint(fwm_s, ff):
                return sum(fwm_s.xs[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.land_constraints[ff]
            fwm_s.c1 = Constraint(fwm_s.farm_ids, rule=land_constraint)


            def water_constraint(fwm_s, ff):
                return sum(fwm_s.xs[i]*fwm_s.nirs[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.water_constraints[ff]
            fwm_s.c2 = Constraint(fwm_s.farm_ids, rule=water_constraint)

            logging.info(f'I have successfully constructed pyomo model for month {month} year {year}.')

            pdb.set_trace()

            ## C.2.c Creating and running the solver:
            try:
                # opt = SolverFactory("ipopt", executable=ipopt_path, solver_io='nl')
                opt = SolverFactory("ipopt", solver_io='nl')
                results = opt.solve(fwm_s, keepfiles=False, tee=True)
                print(results.solver.termination_condition)
            except:
                logging.info(f'Pyomo model solve has failed for month {month} year {year}.')
                return

            logging.info(f'I have successfully solved pyomo model for month {month} year {year}.')

            ## D.1. Storing main model outputs:
            result_xs = dict(fwm_s.xs.get_values())

            # JY store results into a pandas dataframe
            results_pd = data_profit
            results_pd = results_pd.assign(calc_area=result_xs.values())
            results_pd = results_pd.assign(nir=nirs.values())
            results_pd['calc_water_demand'] = results_pd['calc_area'] * results_pd['nir'] / 25583.64
            results_pivot = pd.pivot_table(results_pd, index=['nldas'], values=['calc_water_demand'], aggfunc=np.sum) #JY demand is order of magnitude low, double check calcs

            # JY export results to csv
            results_pd = results_pd[['nldas', 'crop','calc_area']]
            # results_pd.to_csv(output_path+'/abm_results_'+ str(year))
            results_pd.to_csv(output_dir+'/abm_results_'+ str(year))

            # read a sample water demand input file
            file = code_path + '/RCP8.5_GCAM_water_demand_1980_01_copy.nc'
            with netCDF4.Dataset(file, 'r') as nc:
                # for key, var in nc.variables.items():
                #     print(key, var.dimensions, var.shape, var.units, var.long_name, var._FillValue)

                lat = nc['lat'][:]
                lon = nc['lon'][:]
                demand = nc['totalDemand'][:]

            # read NLDAS grid reference file
            df_grid = pd.read_csv(code_path+'/NLDAS_Grid_Reference.csv')

            df_grid = df_grid[['CENTERX', 'CENTERY', 'NLDAS_X', 'NLDAS_Y', 'NLDAS_ID']]

            df_grid = df_grid.rename(columns={"CENTERX": "longitude", "CENTERY": "latitude"})
            df_grid['longitude'] = df_grid.longitude + 360

            mesh_lon, mesh_lat = np.meshgrid(lon, lat)
            df_nc = pd.DataFrame({'lon':mesh_lon.reshape(-1,order='C'),'lat':mesh_lat.reshape(-1,order='C')})
            df_nc['NLDAS_ID'] = ['x'+str(int((row['lon']-235.0625)/0.125+1))+'y'+str(int((row['lat']-25.0625)/0.125+1)) for _,row in df_nc.iterrows()] 
            df_nc['totalDemand'] = 0

            # use NLDAS_ID as index for both dataframes
            df_nc = df_nc.set_index('NLDAS_ID',drop=False)
            try:
                results_pivot = results.pivot.set_index('nldas',drop=False)
            except KeyError:
                pass

            # read ABM values into df_nc basing on the same index
            df_nc.loc[results_pivot.index,'totalDemand'] = results_pivot.calc_water_demand.values

            # set output dir
            path = self.config.get('simulation.output_path') + '/demand'
            logging.info('Outputting demand files to: ' + path)

            for month in months:
                # str_year = str(year)
                # new_fname = output_path+'/RCP8.5_GCAM_water_demand_'+ str_year + '_' + month + '.nc' # define ABM demand input directory
                # new_fname = path + '/demand_' + str_year + '_' + month + '.nc' # define ABM demand input directory
                new_fname = f"{output_dir}{name}_farmer_abm_demand_{year}_{month}.nc"
                shutil.copyfile(file, new_fname)
                demand_ABM = df_nc.totalDemand.values.reshape(len(lat),len(lon),order='C')
                with netCDF4.Dataset(new_fname,'a') as nc:
                    nc['totalDemand'][:] = np.ma.masked_array(demand_ABM,mask=nc['totalDemand'][:].mask)

            logging.info(f'I have successfully written out new demand files for month {month} year {year}.')
            self.processed_years.append(year)
        except Exception as e:
            logging.exception(str(e))
        
        logging.info('Done running.\n')
