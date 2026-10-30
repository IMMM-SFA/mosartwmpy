## create_grand_parameters.py

This utility method generates the four dam/reservoir related input files expected by `mosartwmpy`:
* `grand_reservoir_parameters.nc` - dam/reservoir physical and behavioral parameters (also writable as `.parquet` or `.csv` by changing the output file extension)
* `grand_average_monthly_flow.parquet` - mean monthly flow across the reservoir during the expected simulation period
* `grand_average_monthly_demand.parquet` - mean monthly demand on the reservoir's water during the expected simulation period
* `grand_dependency_database.parquet` - mapping between GRanD ID and grid cell IDs allowed to extract water

Several datasets are required to perform this operation:
* GRanD reservoir shapefiles, i.e. [GRanD_v1_3](http://globaldamwatch.org/grand/)
* Output data from a previous `mosartwmpy` simulation, with water management disabled, covering the desired simulation time period
* Monthly demand input that will be used for the simulation
* The `mosartwmpy` grid (river network) file
* Elevation data covering the `mosartwmpy` domain in parquet format (this data can be upscaled within the utility method if necessary); for instance from [HYDROSHEDS](https://www.hydrosheds.org/downloads) void-filled elevation
* ISTARF dataset with the data-driven reservoir coefficients, i.e. [ISTARF v0.0.1](https://zenodo.org/record/4602277)

Note that the reservoir parameters and dependency database provided in the tutorial are reasonably robust for a 1/8 degree grid --
for most use cases it would be sufficient to simply update the mean flow and demand files as appropriate to your simulation.

### minimum storage

The reservoir parameter file includes a `CAP_MIN` column holding the minimum (dead) storage in million m<sup>3</sup>,
written as `--minimum-storage-fraction` (default `0.1`) times `CAP_MCM`.
GRanD ships a field of the same name meaning a lower bound estimate of the reported *total* capacity, which is not what
`mosartwmpy` expects here, so this utility overwrites it. Adjust individual values afterward where better information is available.

Because the fraction is uniform, any per-reservoir value set by hand afterward is lost the next time this
utility runs. The v1.0 parameter set published with the sample input carries 31 such values, curated from
reported acre-foot dead storages; the other 1832 reservoirs sit at `0.1 * CAP_MCM`. Those 31 are recorded in
[`curated_cap_min.csv`](curated_cap_min.csv) (`GRAND_ID`, `DAM_NAME`, `RES_NAME`, `CAP_MCM`, `CAP_MIN`) so they
can be reapplied after regenerating:

```python
import pandas as pd

parameters = pd.read_csv('grand_reservoir_parameters.csv').set_index('GRAND_ID')
curated = pd.read_csv('curated_cap_min.csv').set_index('GRAND_ID')
parameters.loc[curated.index, 'CAP_MIN'] = curated['CAP_MIN']
```

Thirteen of the 31 are zero. `mosartwmpy` treats a non-positive `CAP_MIN` as absent and falls back to
`reservoir_runoff_capacity_parameter` times capacity, so those reservoirs get the 10% floor rather than
being allowed to draw down to empty; the zeros are kept here to record what the published file contains.
The same values are preserved in the MSD-LIVE sample input (see `sample_input` in
[`data_manifest.yaml`](../data_manifest.yaml)), which remains the canonical source for the full parameter set.

Once the necessary data has been collected, run the utility with the `create_grand_parameters` that is installed along with `mosartwmpy`.
This script will ask for the locations of the datasets and desired output locations.

Optionally, examine the reservoir placement output file with a GIS utility (for instance QGIS), manually update the placement of problem reservoirs, and then rerun this script with a corrections file specified.

For more complete control of the script, examine the [method signature](create_grand_parameters.py) and invoke directly from python.