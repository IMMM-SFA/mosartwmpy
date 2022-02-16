from benedict.dicts import benedict
import netCDF4
import logging
import numpy as np
import pandas as pd
import pickle
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pyutilib.subprocess.GlobalData
import shutil
import sys
import xarray as xr

from mosartwmpy import model


class FarmerABM:

    def __init__(self, model: model):
        self.model = model
        self.config = model.config 
        self.processed_years = []

        self.dependent_cell_index = self.config.get('water_management.reservoirs.dependencies.variables.dependent_cell_index')
        self.grid_cell_id = next((o for o in self.config.get('simulation.grid_output') if o.get('variable', '').casefold() == 'id'), None).get('name')
        self.grid_cell_supply = next((o for o in self.config.get('simulation.output') if o.get('variable', '').casefold() == 'grid_cell_supply'), None).get('name')
        self.latitude = self.config.get('grid.latitude')
        self.longitude = self.config.get('grid.longitude')
        self.mu = self.config.get('water_management.demand.farmer_abm.mu', 0.2)
        self.reservoir_grid_index = self.config.get('water_management.reservoirs.parameters.variables.reservoir_grid_index')
        self.reservoir_id = self.config.get('water_management.reservoirs.parameters.variables.reservoir_id')
        self.reservoir_storage = next((o for o in self.config.get('simulation.output') if o.get('variable', '').casefold() == 'reservoir_storage'), None).get('name')
        self.runoff_land = next((o for o in self.config.get('simulation.output') if o.get('variable', '').casefold() == 'runoff_land'), None).get('name')


    def calc_demand(self):
        code_path = self.config.get('water_management.demand.farmer_abm.path')
        GCS_significant_digits = 4
        month = '1'
        year = self.model.current_time.year

        logging.info('code path: ' + code_path)

        pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False
        output_dir = f"{self.config.get('simulation.output_path')}/demand/"
        months = ['01','02','03','04','05','06','07','08','09','10','11','12']

        # Check that we haven't already performed the farmer ABM calculation.
        if year in self.processed_years:
            logging.debug('Already performed farmer ABM calculations for ' + year + '. Files are in ' + output_dir)
            return

        logging.debug('sys version: ' + str(sys.version_info))
        logging.debug('pandas version: ' + pd.__version__)
        try:
            with open(code_path+'/water_constraints_by_farm_pyt278.p', 'rb') as fp:
                water_constraints_by_farm = pickle.load(fp)
            water_constraints_by_farm = dict.fromkeys(water_constraints_by_farm, 9999999999)

            ## Read in Water Availability Files from MOSART-PMP
            # If year is before 1950 (warm-up period), use the external baseline water demand files
            # TODO: add warmup period. If warmup period is <1 execute this block
            if year < 1950:
                # Use baseline water demand data for warmup period
                water_constraints_by_farm = pd.read_csv(code_path+'/historic_avail_bias_correction_20201102.csv') 
                water_constraints_by_farm = water_constraints_by_farm[['NLDAS_ID','sw_irrigation_vol']].reset_index()
                water_constraints_by_farm = water_constraints_by_farm['sw_irrigation_vol'].to_dict()
            # For first year of ABM, use baseline water demand data
            elif year == 1950:
                water_constraints_by_farm = pd.read_csv(code_path+'/historic_avail_bias_correction_20201102.csv')
                water_constraints_by_farm = water_constraints_by_farm[['NLDAS_ID','sw_irrigation_vol']].reset_index()
                water_constraints_by_farm = water_constraints_by_farm['sw_irrigation_vol'].to_dict()
            else:
                dependency_database_path = self.config.get('water_management.reservoirs.dependencies.path')
                reservoir_parameter_path = self.config.get('water_management.reservoirs.parameters.path')
                simulation_output_path = 'output/istarf_validation/istarf_validation_1982*.nc'
                nldas_path = code_path + '/nldas.txt'
                nldas_ids_path = code_path + '/nldas_ids.p'
                historic_storage_path = code_path + '/hist_dependent_storage.csv'
                historic_avail_bias_path = code_path + '/hist_avail_bias_correction_20201102.csv'
                historic_avail_bias_live_path = code_path + '/hist_avail_bias_correction_live.csv'

                # relationships between Grid Cells and reservoirs they can consume from (many to many)
                dependency_database = pd.read_parquet(dependency_database_path)

                # determines which grid cells the reservoirs are located in (one to one)
                reservoir_parameters = xr.open_dataset(reservoir_parameter_path)[[self.reservoir_id, self.reservoir_grid_index]].to_dataframe()

                simulation_output = xr.open_mfdataset(simulation_output_path)[[
                    self.grid_cell_id, self.reservoir_storage, self.grid_cell_supply, self.runoff_land
                ]].mean('time').to_dataframe().reset_index()

                # not in the ipynb
                if year == '1951':
                    historic_avail_bias = pd.read_csv(historic_avail_bias_path)
                    historic_avail_bias['WRM_SUPPLY_acreft_prev'] = historic_avail_bias['WRM_SUPPLY_acreft_OG']
                else:
                    historic_avail_bias = pd.read_csv(historic_avail_bias_live_path)

                historic_storage = pd.read_csv(historic_storage_path)

                with open(nldas_ids_path, 'rb') as fp:
                    nldas_ids = pickle.load(fp)

                nldas = pd.read_csv(nldas_path)

                # merge the dependencies with the reservoir grid cells
                dependency_database = dependency_database.merge(reservoir_parameters, how='left', on=self.reservoir_id).rename(columns={self.reservoir_grid_index: 'RESERVOIR_CELL_INDEX'})

                # Merge the dependency database with the mean storage at reservoir locations, and aggregate per grid cell
                abm_data = dependency_database.merge(simulation_output[[
                    self.grid_cell_id, self.reservoir_storage
                ]], how='left', left_on='RESERVOIR_CELL_INDEX', right_on=self.grid_cell_id).groupby(self.dependent_cell_index, as_index=False)[[self.reservoir_storage]].sum().rename(
                    columns={self.reservoir_storage: 'STORAGE_SUM'}
                )

                # Merge in the mean supply and mean channel outflow from the simulation results per grid cell
                abm_data[[
                    self.grid_cell_supply, self.runoff_land
                ]] =  abm_data[[self.dependent_cell_index]].merge(simulation_output[[
                    self.grid_cell_id, self.grid_cell_supply, self.runoff_land
                ]], how='left', left_on=self.dependent_cell_index, right_on=self.grid_cell_id)[[
                    self.grid_cell_supply, self.runoff_land
                ]]

                # Merge the lat/lon
                abm_data[[
                    self.latitude, self.longitude
                ]] = abm_data[[self.dependent_cell_index]].merge(simulation_output[[
                    self.grid_cell_id, self.latitude, self.longitude
                ]], how='left', left_on=self.dependent_cell_index, right_on=self.grid_cell_id)[[
                    self.latitude, self.longitude
                ]].round(GCS_significant_digits)

                # Merge the NLDAS_ID
                abm_data = nldas[[
                    'CENTERX', 'CENTERY', 'NLDAS_ID'
                ]].merge(abm_data, left_on=['CENTERY', 'CENTERX'], right_on=[self.latitude, self.longitude], how='left')

                # Select only the NLDAS IDs we care about.
                abm_data = abm_data.loc[abm_data.NLDAS_ID.isin(nldas_ids)]

                # Merge historic storage
                abm_data['STORAGE_SUM_OG'] = abm_data[['NLDAS_ID']].merge(historic_storage[[
                    'NLDAS_ID','STORAGE_SUM_OG'
                ]], on='NLDAS_ID', how='left')[['STORAGE_SUM_OG']]

                # Merge bias correction, original supply in acreft, and original channel outflow
                abm_data[[
                    'sw_avail_bias_corr','WRM_SUPPLY_acreft_OG','RIVER_DISCHARGE_OVER_LAND_LIQ_OG'
                ]] = abm_data[['NLDAS_ID']].merge(historic_avail_bias[[
                    'NLDAS_ID','sw_avail_bias_corr','WRM_SUPPLY_acreft_OG','RIVER_DISCHARGE_OVER_LAND_LIQ_OG'
                ]], on='NLDAS_ID', how='left')[[
                    'sw_avail_bias_corr','WRM_SUPPLY_acreft_OG','RIVER_DISCHARGE_OVER_LAND_LIQ_OG'
                ]]

                # Rename original supply
                abm_data['WRM_SUPPLY_acreft_prev'] = abm_data['WRM_SUPPLY_acreft_OG']

                # Zero the missing data.
                abm_data = abm_data.fillna(0)

                # Calculate a "demand factor" for each agent.
                abm_data['demand_factor'] = np.where(
                    abm_data['STORAGE_SUM_OG'] > 0,
                    abm_data['STORAGE_SUM'] / abm_data['STORAGE_SUM_OG'],
                    np.where(
                        abm_data['RIVER_DISCHARGE_OVER_LAND_LIQ_OG'] >= 0.1,
                        abm_data[self.runoff_land] / abm_data['RIVER_DISCHARGE_OVER_LAND_LIQ_OG'],
                        1
                    )
                )

                abm_data['WRM_SUPPLY_acreft_newinfo'] = abm_data['demand_factor'] * abm_data['WRM_SUPPLY_acreft_OG']

                abm_data['WRM_SUPPLY_acreft_updated'] = ((1 - self.mu) * abm_data['WRM_SUPPLY_acreft_prev']) + (self.mu * abm_data['WRM_SUPPLY_acreft_newinfo'])

                abm_data['WRM_SUPPLY_acreft_prev'] = abm_data['WRM_SUPPLY_acreft_updated']

                # Update CSV with 'live' data
                # not in the ipynb
                abm_data[['NLDAS_ID','WRM_SUPPLY_acreft_OG','WRM_SUPPLY_acreft_prev','sw_avail_bias_corr','demand_factor','RIVER_DISCHARGE_OVER_LAND_LIQ_OG']].to_csv(code_path+'/hist_avail_bias_correction_live.csv')

                abm_data['WRM_SUPPLY_acreft_bias_corr'] = abm_data['WRM_SUPPLY_acreft_updated'] + abm_data['sw_avail_bias_corr']

                water_constraints_by_farm = abm_data.reset_index()['WRM_SUPPLY_acreft_bias_corr'].to_dict()
                logging.info(f'Converted units df for month {month} year {year}')


            logging.info(f'Loaded water availability files for month {month} year {year}.')

            ## Read in PMP calibration files
            data_file=pd.ExcelFile(code_path+"/MOSART_WM_PMP_inputs_20201028.xlsx")
            data_profit = data_file.parse("Profit")
            water_nirs=data_profit["nir_corrected"]
            nirs=dict(water_nirs)

            logging.info(f'Loaded PMP calibration files for month {month} year {year}.')

            # TODO: stop hardcoding this
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

            logging.info(f'Loaded constructed model indices, constraints for month {month} year {year}.')

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

            logging.info(f'Successfully constructed pyomo model for month {month} year {year}.')

            ## C.2.c Creating and running the solver:
            try:
                opt = SolverFactory("ipopt", solver_io='nl')
                results = opt.solve(fwm_s, keepfiles=False, tee=True)
                print(results.solver.termination_condition)
                logging.info(f'I have successfully solved pyomo model for month {month} year {year}.')
            except:
                logging.info(f'Pyomo model solve has failed for month {month} year {year}.')
                return

            ## D.1. Storing main model outputs:
            result_xs = dict(fwm_s.xs.get_values())

            # Store results into a pandas dataframe
            results_pd = data_profit
            results_pd = results_pd.assign(calc_area=result_xs.values())
            results_pd = results_pd.assign(nir=nirs.values())
            results_pd['calc_water_demand'] = results_pd['calc_area'] * results_pd['nir'] / 25583.64
            results_pivot = pd.pivot_table(results_pd, index=['nldas'], values=['calc_water_demand'], aggfunc=np.sum) #JY demand is order of magnitude low, double check calcs

            # Export results to csv
            results_pd = results_pd[['nldas', 'crop','calc_area']]
            results_pd.to_csv(output_dir+'/abm_results_'+ str(year))

            # read a sample water demand input file
            file = code_path + '/RCP8.5_GCAM_water_demand_1980_01_copy.nc'
            with netCDF4.Dataset(file, 'r') as nc:
                lat = nc[self.latitude][:]
                lon = nc[self.longitude][:]
                demand = nc['totalDemand'][:]

            # read NLDAS grid reference file
            df_grid = pd.read_csv(code_path+'/NLDAS_Grid_Reference.csv')

            df_grid = df_grid[['CENTERX', 'CENTERY', 'NLDAS_X', 'NLDAS_Y', 'NLDAS_ID']]

            df_grid = df_grid.rename(columns={"CENTERX": "longitude", "CENTERY": "latitude"})
            df_grid['longitude'] = df_grid.longitude + 360

            mesh_lon, mesh_lat = np.meshgrid(lon, lat)
            df_nc = pd.DataFrame({self.longitude:mesh_lon.reshape(-1,order='C'),self.latitude:mesh_lat.reshape(-1,order='C')})
            df_nc['NLDAS_ID'] = ['x'+str(int((row[self.longitude]-235.0625)/0.125+1))+'y'+str(int((row[self.latitude]-25.0625)/0.125+1)) for _,row in df_nc.iterrows()] 
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
                new_fname = f"{output_dir}{self.model.name}_farmer_abm_demand_{year}_{month}.nc"
                shutil.copyfile(file, new_fname)
                demand_ABM = df_nc.totalDemand.values.reshape(len(lat),len(lon),order='C')
                with netCDF4.Dataset(new_fname,'a') as nc:
                    nc['totalDemand'][:] = np.ma.masked_array(demand_ABM,mask=nc['totalDemand'][:].mask)

            logging.info(f'Written out new demand files for month {month} year {year}.')
            self.processed_years.append(year)
        except Exception as e:
            logging.exception(str(e))
        
        logging.info('Done running. \n')
