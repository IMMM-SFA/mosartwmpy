## create_grand_parameters.py

This utility method generates the four dam/reservoir related input files expected by `mosartwmpy`:
* `grand_reservoir_parameters.nc` - dam/reservoir physical and behavioral parameters
* `grand_average_monthly_flow.parquet` - mean monthly flow across the reservoir during the expected simulation period
* `grand_average_monthly_demand.parquet` - mean monthly demand on the reservoir's water during the expected simulation period
* `grand_dependency_database.parquet` - mapping between GRanD ID and grid cell IDs allowed to extract water

Several datasets are required to perform this operation:
* GRanD reservoir shapefiles, i.e. [GRanD_v1_3](https://todo)
* Output data from a previous `mosartwmpy` simulation, with water management disabled, covering the desired simulation time period
* Monthly demand input that will be used for the simulation
* The `mosartwmpy` grid (river network) file
* Elevation data covering the `mosartwmpy` domain in parquet format (this data can be upscaled within the utility method if necessary)
* ISTARF dataset with the data-driven reservoir coefficients, i.e. [ISTARF v1.0](https://todo)

Note that the reservoir parameters and dependency database provided in the tutorial are reasonably robust for a 1/8 degree grid --
for most use cases it would be sufficient to simply update the mean flow and demand files as appropriate to your simulation.

Once the necessary data has been collected, run the utility with the `create_grand_parameters` that is installed along with `mosartwmpy`.
This script will ask for the locations of the datasets and desired output locations.
For more complete control of the script, examine the [method signature](create_grand_parameters.py) and invoke directly from python.