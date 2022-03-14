import logging
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pyutilib.subprocess.GlobalData
from timeit import default_timer as timer
import xarray as xr

from mosartwmpy import model
from mosartwmpy.utilities.pretty_timer import pretty_timer

class FarmerABM:

    def __init__(self, model: model):
        self.model = model
        self.config = model.config 
        self.processed_years = []

        self.demand = self.config.get('water_management.demand.demand')
        self.dependent_cell_index = self.config.get('water_management.reservoirs.dependencies.variables.dependent_cell_index')
        self.grid_cell_id = next((o for o in self.config.get('simulation.grid_output') if o.get('variable', '').casefold() == 'id'), None).get('name')
        self.grid_cell_supply = next((o for o in self.config.get('simulation.output') if o.get('variable', '').casefold() == 'grid_cell_supply'), None).get('name')
        self.latitude = self.config.get('grid.latitude')
        self.longitude = self.config.get('grid.longitude')
        self.mu = self.config.get('water_management.demand.farmer_abm.mu', 0.2)
        self.nldas_id = next((o for o in self.config.get('simulation.grid_output') if o.get('variable', '').casefold() == 'nldas_id'), None).get('name')
        self.reservoir_grid_index = self.config.get('water_management.reservoirs.parameters.variables.reservoir_grid_index')
        self.reservoir_id = self.config.get('water_management.reservoirs.parameters.variables.reservoir_id')
        self.reservoir_storage = next((o for o in self.config.get('simulation.output') if o.get('variable', '').casefold() == 'reservoir_storage'), None).get('name')
        self.runoff_land = next((o for o in self.config.get('simulation.output') if o.get('variable', '').casefold() == 'runoff_land'), None).get('name')
        self.time = self.config.get('water_management.demand.time')


    def calc_demand(self):
        t = timer()
        month = '01'
        year = self.model.current_time.year
        # Conversion from acre-feet/year to cubic meters/sec (the demand units that MOSART-WM takes in).
        ACREFTYEAR_TO_CUBICMSEC = 25583.64
        pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

        # All file paths.
        dependency_database_path = self.config.get('water_management.reservoirs.dependencies.path')
        historic_storage_supply_path = f"{self.config.get('water_management.demand.farmer_abm.historic_storage_supply.path')}"
        land_water_constraints_by_farm_live_path = f"{self.config.get('water_management.demand.farmer_abm.land_water_constraints_live.path')}"
        land_water_constraints_by_farm_path = self.config.get('water_management.demand.farmer_abm.land_water_constraints.path')
        mosart_wm_pmp_path = f"{self.config.get('water_management.demand.farmer_abm.mosart_wm_pmp.path')}"
        output_dir = f"{self.config.get('water_management.demand.output.path')}"
        reservoir_parameter_path = self.config.get('water_management.reservoirs.parameters.path')
        simulation_output_path = f"{self.config.get('simulation.output_path')}/{self.config.get('simulation.name')}/{self.config.get('simulation.name')}_{year}_*.nc"

        # Check we haven't already performed the farmer ABM calculation.
        if year in self.processed_years:
            logging.debug(f"Already performed farmer ABM calculations for {year}. Files are in {output_dir}.")
            return

        try:
            land_water_constraints_by_farm = pd.read_parquet(land_water_constraints_by_farm_path)

            # If warm-up period is < 1, use the external baseline water demand files
            if self.config.get('water_management.demand.farmer_abm.warmup_period') < 1:
                water_constraints_by_farm = land_water_constraints_by_farm[['NLDAS_ID','sw_irrigation_vol']].reset_index()
                water_constraints_by_farm = water_constraints_by_farm['sw_irrigation_vol'].to_dict()
            else:
                historic_storage_supply = pd.read_parquet(historic_storage_supply_path)

                # Relationships between grid cells and reservoirs they can consume from (many to many)
                dependency_database = pd.read_parquet(dependency_database_path)

                # Determines which grid cells the reservoirs are located in (one to one)
                reservoir_parameters = xr.open_dataset(reservoir_parameter_path)[[self.reservoir_id, self.reservoir_grid_index]].to_dataframe()

                # Get mosartwmpy output
                simulation_output = xr.open_mfdataset(simulation_output_path)[[
                    self.grid_cell_id, self.reservoir_storage, self.grid_cell_supply, self.runoff_land, self.nldas_id
                ]].mean('time').to_dataframe().reset_index()

                # Merge the dependencies with the reservoir grid cells
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

                # Merge in NLDAS ID from simulation output
                abm_data = simulation_output[[
                    self.grid_cell_supply, self.nldas_id
                ]].merge(abm_data, left_on=self.grid_cell_supply, right_on=self.dependent_cell_index, how='left')
                    
                # Merge bias correction, original supply in acreft, historic storage, and original channel outflow
                abm_data[[
                    'sw_avail_bias_corr','WRM_SUPPLY_acreft_OG','RIVER_DISCHARGE_OVER_LAND_LIQ_OG','STORAGE_SUM_OG'
                ]] = abm_data[[self.nldas_id]].merge(historic_storage_supply[[
                    'NLDAS_ID','sw_avail_bias_corr','WRM_SUPPLY_acreft_OG','RIVER_DISCHARGE_OVER_LAND_LIQ_OG','STORAGE_SUM_OG'
                ]], left_on=self.nldas_id, right_on='NLDAS_ID', how='left')[[
                    'sw_avail_bias_corr','WRM_SUPPLY_acreft_OG','RIVER_DISCHARGE_OVER_LAND_LIQ_OG','STORAGE_SUM_OG'
                ]]

                # Select only NLDAS_IDs in historic_storage_supply.
                abm_data = abm_data.loc[abm_data[self.nldas_id].isin(historic_storage_supply[self.nldas_id])]

                # Rename original supply.
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

                # TODO: what is the live data for and do we need it?
                # Update parquet with 'live' data.
                abm_data[[self.nldas_id,'WRM_SUPPLY_acreft_OG','WRM_SUPPLY_acreft_prev','sw_avail_bias_corr','demand_factor','RIVER_DISCHARGE_OVER_LAND_LIQ_OG']].to_parquet(land_water_constraints_by_farm_live_path)

                abm_data['WRM_SUPPLY_acreft_bias_corr'] = abm_data['WRM_SUPPLY_acreft_updated'] + abm_data['sw_avail_bias_corr']

                water_constraints_by_farm = abm_data.reset_index()['WRM_SUPPLY_acreft_bias_corr'].to_dict()
                logging.info(f"Converted units dataframe for month {month} year {year}")

            logging.info(f"Loaded water availability files for month {month} year {year}.")

            # Read in PMP calibration files.
            data_profit = pd.read_parquet(mosart_wm_pmp_path)
            nirs = data_profit['nir_corrected'].to_dict()
            gammas = data_profit['gammas'].to_dict()
            net_prices = data_profit['net_prices'].to_dict()

            logging.info(f"Loaded PMP calibration files for month {month} year {year}.")

            # Number of crop and NLDAS ID combinations.
            ids = range(len(data_profit)) 
            # Number of farm agents / NLDAS IDs.
            farm_ids = range(len(pd.unique(data_profit['nldas'])))

            land_constraints_by_farm = land_water_constraints_by_farm['LAND_CONSTRAINTS_BY_FARM'].to_dict()

            crop_ids_by_farm = data_profit.drop(columns='index').reset_index().groupby(by='nldas')['index'].apply(list)
            crop_ids_by_farm.set_axis(range(0, len(crop_ids_by_farm)), inplace = True)
            crop_ids_by_farm = crop_ids_by_farm.to_dict()

            # Initialize start values to zero.
            x_start_values=dict(enumerate([0.0]*3))

            logging.info(f"Loaded constructed model indices and constraints for month {month} year {year}.")

            # 2st stage: Quadratic model included in JWP model simulations.
            # Construct model inputs.
            fwm_s = ConcreteModel()
            fwm_s.ids = Set(initialize=ids)
            fwm_s.farm_ids = Set(initialize=farm_ids)
            fwm_s.crop_ids_by_farm = Set(fwm_s.farm_ids, initialize=crop_ids_by_farm)
            fwm_s.net_prices = Param(fwm_s.ids, initialize=net_prices, mutable=True)
            fwm_s.gammas = Param(fwm_s.ids, initialize=gammas, mutable=True)
            fwm_s.land_constraints = Param(fwm_s.farm_ids, initialize=land_constraints_by_farm, mutable=True)
            fwm_s.water_constraints = Param(fwm_s.farm_ids, initialize=water_constraints_by_farm, mutable=True) #JY here need to read calculate new water constraints
            fwm_s.xs = Var(fwm_s.ids, domain=NonNegativeReals, initialize=x_start_values)
            fwm_s.nirs = Param(fwm_s.ids, initialize=nirs, mutable=True)


            # 2nd stage model: Construct functions.
            def obj_fun(fwm_s):
                return 0.00001*sum(sum((fwm_s.net_prices[i] * fwm_s.xs[i] - 0.5 * fwm_s.gammas[i] * fwm_s.xs[i] * fwm_s.xs[i]) for i in fwm_s.crop_ids_by_farm[f]) for f in fwm_s.farm_ids)
            fwm_s.obj_f = Objective(rule=obj_fun, sense=maximize)


            def land_constraint(fwm_s, ff):
                return sum(fwm_s.xs[i] for i in fwm_s.crop_ids_by_farm[ff]) <= fwm_s.land_constraints[ff]
            fwm_s.c1 = Constraint(fwm_s.farm_ids, rule=land_constraint)


            def water_constraint(fwm_s, ff):
                return sum(fwm_s.xs[i]*fwm_s.nirs[i] for i in fwm_s.crop_ids_by_farm[ff]) <= fwm_s.water_constraints[ff]
            fwm_s.c2 = Constraint(fwm_s.farm_ids, rule=water_constraint)


            logging.info(f"Constructed pyomo model for month {month} year {year}.")

            # Create and run the solver.
            try:
                opt = SolverFactory("ipopt", solver_io='nl')
                results = opt.solve(fwm_s, keepfiles=False, tee=True)
                print(results.solver.termination_condition)
                logging.info(f"Solved pyomo model for month {month} year {year}.")
            except:
                logging.info(f"Pyomo model solve has failed for month {month} year {year}.")
                return

            # Store main model outputs.
            result_xs = dict(fwm_s.xs.get_values())

            # Store results in a pandas dataframe.
            results_pd = data_profit
            results_pd = results_pd.assign(calc_area=result_xs.values())
            results_pd = results_pd.assign(nir=nirs.values())
            results_pd['calc_water_demand'] = results_pd['calc_area'] * results_pd['nir'] / ACREFTYEAR_TO_CUBICMSEC
            # TODO: Address the comment below.
            results_pivot = pd.pivot_table(results_pd, index=['nldas'], values=['calc_water_demand'], aggfunc=np.sum) #JY demand is order of magnitude low, double check calcs

            # TODO: Would we rather have parquet?
            # Export results to CSV.
            results_pd = results_pd[['nldas', 'crop', 'calc_area']]
            results_pd.to_csv(output_dir+'/farmer_abm_'+ str(year))

            # Construct a DataFrame with all NLDAS IDs.
            demand_per_nldas_id = pd.DataFrame(self.model.grid.nldas_id).rename(columns={0:self.nldas_id})
            demand_per_nldas_id[self.demand] = 0

            # Use NLDAS_ID as the index and merge ABM demand.
            demand_per_nldas_id = demand_per_nldas_id.set_index(self.nldas_id,drop=False)
            demand_per_nldas_id.loc[results_pivot.index,self.demand] = results_pivot.calc_water_demand.values

            # Convert pandas DataFrame to xarray Dataset to more easily output to a NetCDF.
            demand_ABM = demand_per_nldas_id.totalDemand.values.reshape(
                len(self.model.grid.unique_latitudes), len(self.model.grid.unique_longitudes),
                order='C'
            )
            demand_ABM = xr.Dataset(
                data_vars={
                    self.demand:([self.time, self.latitude, self.longitude], np.array([demand_ABM]))
                },
                coords={
                    self.longitude: ([self.longitude], self.model.grid.unique_longitudes),
                    self.latitude: ([self.latitude], self.model.grid.unique_latitudes),
                    self.time: ([self.time], [np.datetime64(f"{year}-01")]),
                }
            )
            logging.info(f"Outputting demand file to: {output_dir}.")
            demand_ABM.to_netcdf(f"{output_dir}/{self.config.get('simulation.name')}_farmer_abm_demand_{year}.nc")
            logging.info(f"Wrote new demand files for month {month} year {year}.")

            self.processed_years.append(year)
        except Exception as e:
            logging.exception(str(e))
        
        logging.info(f"Ran farmer ABM in {pretty_timer(timer() - t)}.")