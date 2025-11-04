# MOSARTWMPY - Code Structure and Logic Documentation

## Overview

**mosartwmpy** is a Python implementation of MOSART-WM (Model Of Scale Adaptive River Transports - Water Management), a hydrologic routing and reservoir management model. It simulates water movement through river networks and manages water resources via reservoir operations.

**Key Capabilities:**
- Multi-scale water routing (hillslope → tributary → main channel)
- Reservoir management and water demand satisfaction
- Integration with Earth system models via BMI (Basic Model Interface)
- Agent-based modeling for agricultural water demand

**Published:** Journal of Open Source Software - DOI: 10.21105/joss.03221

## Project Statistics

- **Python LOC:** ~5,322 lines
- **State Variables:** ~200 arrays
- **Grid Parameters:** 60+ spatial fields
- **Supported Python:** 3.9 - 3.12
- **Performance:** Numba JIT-compiled core routing algorithms

---

## Directory Structure

```
mosartwmpy/
├── mosartwmpy/                    # Main package
│   ├── model.py                   # BMI Model orchestrator (445 lines)
│   ├── config_defaults.yaml       # Default configuration
│   ├── input_output_variables.py  # Variable metadata for BMI
│   │
│   ├── config/                    # Configuration system
│   │   ├── config.py              # YAML config loading/merging
│   │   └── parameters.py          # Physical constants
│   │
│   ├── grid/                      # Spatial domain
│   │   └── grid.py                # Grid class: topology + parameters
│   │
│   ├── state/                     # Model state
│   │   └── state.py               # State class: ~200 state arrays
│   │
│   ├── input/                     # Input handling
│   │   ├── runoff.py              # Load runoff forcing
│   │   └── demand.py              # Load water demand
│   │
│   ├── output/                    # Output handling
│   │   └── output.py              # Write NetCDF output/restart files
│   │
│   ├── update/                    # Main simulation logic
│   │   └── update.py              # Timestep advancement
│   │
│   ├── hillslope/                 # Hillslope routing
│   │   └── routing.py             # Overland flow calculations
│   │
│   ├── subnetwork/                # Tributary channel routing
│   │   ├── routing.py             # Subnetwork flow dynamics
│   │   ├── irrigation.py          # Water extraction
│   │   └── state.py               # State updates
│   │
│   ├── main_channel/              # Main river channel routing
│   │   ├── routing.py             # Channel flow orchestration
│   │   ├── kinematic_wave.py      # Kinematic wave solver
│   │   ├── irrigation.py          # Water extraction
│   │   └── state.py               # State updates
│   │
│   ├── reservoirs/                # Water management
│   │   ├── release.py             # Reservoir release logic
│   │   ├── regulation.py          # Storage regulation
│   │   ├── istarf.py              # Data-driven release rules
│   │   ├── state.py               # Reservoir initialization
│   │   └── grid.py                # Load reservoir parameters
│   │
│   ├── farmer_abm/                # Agent-based model
│   │   └── farmer_abm.py          # Farmer demand optimization
│   │
│   ├── plotting/                  # Visualization
│   │   └── plot.py                # Plot variables and reservoirs
│   │
│   ├── utilities/                 # Helper tools
│   │   ├── download_data.py       # Download sample datasets
│   │   ├── create_grand_parameters.py  # Process GRanD database
│   │   ├── timing.py              # Performance profiling
│   │   └── ...
│   │
│   └── tests/                     # Unit tests
│
├── docs/                          # Sphinx documentation
├── notebooks/                     # Tutorial Jupyter notebooks
├── validation/                    # Validation scripts
└── launch.py                      # Simple launch script
```

---

## Core Components

### 1. Model Class (`model.py`)

**Purpose:** Main orchestrator implementing BMI interface

**Key Responsibilities:**
- Initialize simulation from config file
- Advance model timestep (`update()`)
- Provide BMI getter/setter methods for coupled modeling
- Coordinate I/O operations
- Manage visualization

**Key Attributes:**
```python
config           # Configuration (Benedict dict)
grid             # Grid object (spatial domain + parameters)
state            # State object (~200 state arrays)
current_time     # Current simulation datetime
parameters       # Physical parameters (Manning coefficients, etc.)
farmerABM        # Optional farmer agent-based model
mask             # Active grid cell mask
output_buffer    # Output accumulator for averaging
```

**Key Methods:**
```python
initialize(config_file)              # Setup simulation
update()                             # Advance one timestep
update_until(time)                   # Run until specified time
finalize()                           # Cleanup and close files
get_value(var_name)                  # BMI: get variable values
set_value(var_name, values)          # BMI: set variable values
plot_variable(var_name, log_scale)   # Visualize current state
```

### 2. Grid Class (`grid/grid.py`)

**Purpose:** Store spatially-varying parameters and network topology

**Key Data Categories:**

**Coordinates & Identifiers:**
```python
id, latitude, longitude
downstream_id          # Index of downstream cell
outlet_id              # Basin outlet identifier
upstream_id            # List of upstream cell indices
```

**Network Topology:**
```python
iterations_main_channel     # Routing subcycles for stability
iterations_subnetwork       # Subnetwork subcycles
total_drainage_area_single  # Drainage area (single direction)
total_drainage_area_multi   # Drainage area (multi-direction)
upstream_cell_count         # Number of upstream cells
land_mask, mosart_mask      # Active cell masks
```

**Physical Parameters:**
```python
# Manning roughness coefficients
hillslope_manning, subnetwork_manning, channel_manning

# Slopes (m/m)
hillslope_slope, subnetwork_slope, channel_slope

# Dimensions
subnetwork_width, channel_width, channel_length
channel_floodplain_width

# Area (m²)
area, land_fraction, drainage_fraction, drainage_density
```

**Reservoir Parameters:**
```python
reservoir_id                      # GRanD database ID
reservoir_grid_index              # Grid cell containing reservoir
reservoir_storage_capacity        # Maximum storage (m³)
reservoir_surface_area            # Surface area (m²)
reservoir_height                  # Dam height (m)
reservoir_use_*                   # Use flags (irrigation, flood control, etc.)
reservoir_streamflow_schedule     # Expected monthly inflow
reservoir_demand_schedule         # Expected monthly demand
grid_index_to_reservoirs_map      # Numba Dict for fast lookup
```

### 3. State Class (`state/state.py`)

**Purpose:** Store all model state variables (~200 arrays)

**Major Variable Groups:**

**Flow & Storage (m³/s, m³):**
```python
flow                           # Total flow
storage                        # Total storage
channel_storage                # Main channel storage
subnetwork_storage             # Tributary storage
delta_storage*                 # Change in storage per timestep
```

**Hillslope Variables (m, m/s):**
```python
hillslope_depth                # Water depth
hillslope_storage              # Water storage
hillslope_surface_runoff       # Surface runoff input
hillslope_subsurface_runoff    # Subsurface runoff input
hillslope_wetland_runoff       # Wetland runoff input
hillslope_overland_flow        # Overland flow velocity
```

**Subnetwork Variables (m³, m/s, m):**
```python
subnetwork_storage             # Storage in tributary channels
subnetwork_discharge           # Discharge rate
subnetwork_depth               # Water depth
subnetwork_flow_velocity       # Flow velocity
subnetwork_cross_section_area  # Cross-sectional area
```

**Main Channel Variables (m³, m/s, m):**
```python
channel_storage                # Storage in main channel
channel_outflow_downstream*    # Outflow to downstream cell
channel_depth                  # Water depth
channel_flow_velocity          # Flow velocity
channel_cross_section_area     # Cross-sectional area
```

**Water Management (m³/s, m³):**
```python
reservoir_storage              # Reservoir storage
reservoir_release              # Reservoir release rate
grid_cell_supply               # Water supply to demand cells
grid_cell_demand_rate          # Water demand rate
grid_cell_deficit              # Unmet demand deficit
grid_cell_unmet_demand         # Accumulated unmet demand
```

**Metadata:**
```python
tracer                         # Liquid/ice type (1=liquid, 2=ice)
euler_mask                     # Active cells for computation
```

### 4. Configuration System (`config/`)

**Config Loading (`config.py`):**
- Merges `config_defaults.yaml` with user-provided `config.yaml`
- Uses Benedict library for hierarchical YAML access
- Supports dotted-key notation: `config.get('simulation.timestep')`

**Key Configuration Sections:**

```yaml
simulation:
  name: "simulation_name"
  start_date: "1981-01-01"
  end_date: "1981-12-31"
  timestep: 10800              # seconds (3 hours default)
  subcycles: 10                # routing subcycles for stability
  routing_iterations: 1        # iterations per subcycle
  log_level: "INFO"
  output_path: "./output"
  output_resolution: "daily"   # daily, monthly, yearly
  output: [list of variables]  # variables to write
  restart_file: null           # path to restart file
  restart_file_frequency: "yearly"

grid:
  path: "./input/domains/mosart_conus_nldas_grid.nc"
  longitude: "lonc"            # longitude variable name
  latitude: "latc"             # latitude variable name
  subdomain: null              # optional: list of coordinates
  unmask_output: false         # write all cells or only active
  variables: {...}             # mapping of field names

runoff:
  read_from_file: true
  path: "./input/runoff/runoff-*.nc"  # supports placeholders
  time: "time"
  longitude: "lon"
  latitude: "lat"
  variables:
    surface_runoff: "QOVER"
    subsurface_runoff: "QDRAI"
    wetland_runoff: "QWETLAND"

water_management:
  enabled: true
  demand:
    read_from_file: true
    path: "./input/demand/demand-*.nc"
    time: "time"
    longitude: "lon"
    latitude: "lat"
    demand: "totalDemand"
    farmer_abm:                # optional: agent-based model
      enabled: false
      ...
  reservoirs:
    enable_istarf: true        # data-driven release rules
    parameters: "./input/reservoirs/reservoirs.nc"
    dependencies: "./input/reservoirs/dependency_database.parquet"
    streamflow: "./input/reservoirs/mean_monthly_reservoir_flow.parquet"
    demand: "./input/reservoirs/mean_monthly_reservoir_demand.parquet"
```

**Physical Parameters (`parameters.py`):**
```python
# Numerical thresholds
tiny_value = 1.0e-14
small_value = 1.0e-10

# Flow parameters
flood_threshold = 0.1           # m
river_depth_minimum = 0.01      # m

# Reservoir parameters
reservoir_regulation_param = 0.85
reservoir_flood_control_param = 0.9

# Irrigation parameters
extraction_maximum_fraction = 0.5
```

---

## Main Logic and Algorithms

### Initialization Flow

```
Model.initialize(config_file)
    ↓
1. Load and merge configuration
    ↓
2. Create Grid from grid file
   - Load domain coordinates
   - Load network topology (downstream_id, upstream_id)
   - Load physical parameters (Manning, slopes, dimensions)
   - Load reservoir parameters (if water_management enabled)
    ↓
3. Create State (allocate ~200 arrays)
   - Initialize to zeros/NaN
   - Set initial conditions from restart file (if provided)
    ↓
4. Initialize water management (if enabled)
   - Load reservoir state
   - Initialize FarmerABM (if configured)
    ↓
5. Apply domain mask
   - Trim grid and state to active cells
   - Improve performance by reducing array size
    ↓
6. Initialize output system
   - Setup output buffer
   - Create output directory structure
```

### Timestep Update Flow

The `Model.update()` method advances the simulation by one timestep through these stages:

```
Model.update()
    ↓
┌─────────────────────────────────────────┐
│ 1. LOAD FORCING DATA (if read_from_file)│
├─────────────────────────────────────────┤
│ - load_runoff()                         │
│   → Read surface/subsurface runoff      │
│   → Convert mm/s to m³/s                │
│   → Assign to hillslope_*_runoff        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. WATER MANAGEMENT (if enabled)        │
├─────────────────────────────────────────┤
│ At month boundary:                      │
│ - load_demand()                         │
│   → Read m³/s from file OR              │
│   → Calculate via farmer_abm            │
│                                         │
│ - reservoir_release()                   │
│   → Compute monthly release targets     │
│   → Apply Biemans (2011) rules          │
│   → Apply ISTARF rules (if enabled)     │
│   → Adjust for flood control            │
│                                         │
│ - Zero supply/demand accumulators       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. ROUTING (multiple subcycles)         │
├─────────────────────────────────────────┤
│ FOR each subcycle:                      │
│                                         │
│   _prepare()                            │
│   ├─ Calculate direct flow              │
│   └─ Sum runoff totals                  │
│                                         │
│   FOR each routing_iteration:           │
│                                         │
│     ┌─ SUBNETWORK ─────────────────┐   │
│     │ FOR each subnetwork_iter:    │   │
│     │                              │   │
│     │   hillslope_routing()        │   │
│     │   ├─ Calculate velocity      │   │
│     │   ├─ Update storage          │   │
│     │   └─ Output to subnetwork    │   │
│     │                              │   │
│     │   subnetwork_irrigation()    │   │
│     │   ├─ Extract for demand      │   │
│     │   └─ Update unmet demand     │   │
│     │                              │   │
│     │   subnetwork_routing()       │   │
│     │   ├─ Manning velocity        │   │
│     │   ├─ Compute discharge       │   │
│     │   ├─ Update storage          │   │
│     │   └─ Output to channel       │   │
│     └──────────────────────────────┘   │
│                                         │
│     [UPSTREAM INTERACTIONS]             │
│     ├─ Route outflow downstream         │
│     └─ Accumulate upstream inflow       │
│                                         │
│     ┌─ MAIN CHANNEL ────────────────┐  │
│     │ FOR each channel_iter:        │  │
│     │                               │  │
│     │   main_channel_irrigation()   │  │
│     │   ├─ Extract for demand       │  │
│     │   └─ Update supply/unmet      │  │
│     │                               │  │
│     │   main_channel_routing()      │  │
│     │   ├─ kinematic_wave_routing() │  │
│     │   ├─ Update depth/velocity    │  │
│     │   └─ Output downstream        │  │
│     └───────────────────────────────┘  │
│                                         │
│     [RESERVOIR REGULATION]              │
│     ├─ regulation()                     │
│     │  ├─ Update storage              │
│     │  └─ Apply min/max constraints   │
│     │                                  │
│     ├─ extraction_regulated_flow()    │
│     │  ├─ Distribute supply           │
│     │  └─ Track deficit               │
│     │                                  │
│     └─ storage_targets()               │
│        └─ Regulate to monthly target   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. OUTPUT                               │
├─────────────────────────────────────────┤
│ - update_output()                       │
│   → Accumulate to buffer                │
│                                         │
│ - [At period end] write_output()        │
│   → Average buffer                      │
│   → Write to NetCDF                     │
│                                         │
│ - check_restart()                       │
│   → Write restart file (if scheduled)   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. CLEANUP                              │
├─────────────────────────────────────────┤
│ - Clear runoff input arrays             │
│ - Increment current_time                │
└─────────────────────────────────────────┘
```

### Core Routing Algorithms

All routing functions use **Numba JIT compilation** for performance and are parallelized with `@nb.jit(parallel=True)`.

#### 1. Hillslope Routing (`hillslope/routing.py`)

**Purpose:** Route water from land surface to tributary channels

**Algorithm:**
```python
def hillslope_routing(hillslope_surface_runoff, hillslope_subsurface_runoff, ...):
    for i in prange(n):  # Parallel loop over grid cells
        # 1. Calculate overland flow velocity (Manning equation)
        depth = hillslope_storage[i] / area[i]
        velocity = (depth^(2/3) * hillslope_slope[i]^(1/2)) / manning[i]

        # 2. Update hillslope storage
        inflow = (surface_runoff[i] + subsurface_runoff[i]) * delta_t
        outflow = velocity * depth * width * delta_t
        hillslope_storage[i] = hillslope_storage[i] + inflow - outflow

        # 3. Output to subnetwork
        subnetwork_inflow[i] = outflow / delta_t
```

**Key Equations:**
- Manning equation: `v = (R^(2/3) * S^(1/2)) / n`
  - `R` = hydraulic radius ≈ depth
  - `S` = slope
  - `n` = Manning roughness coefficient

#### 2. Subnetwork Routing (`subnetwork/routing.py`)

**Purpose:** Route water through tributary channels to main channel

**Algorithm:**
```python
def subnetwork_routing(subnetwork_storage, hillslope_outflow, ...):
    for i in prange(n):
        # 1. Calculate flow velocity (Manning equation)
        if subnetwork_storage[i] > threshold:
            depth = subnetwork_storage[i] / (width[i] * length[i])
            hydraulic_radius = compute_hydraulic_radius(depth, width[i])
            velocity = (hydraulic_radius^(2/3) * slope[i]^(1/2)) / manning[i]

        # 2. Compute discharge
        cross_section = depth * width[i]
        discharge = velocity * cross_section

        # 3. Update storage
        inflow = hillslope_outflow[i] * delta_t
        outflow = discharge * delta_t
        subnetwork_storage[i] += inflow - outflow

        # 4. Output to main channel
        channel_inflow[i] = discharge
```

#### 3. Main Channel Routing (`main_channel/routing.py`)

**Purpose:** Route water through main river channels using kinematic wave

**Algorithm:**
```python
def main_channel_routing(channel_storage, subnetwork_outflow, ...):
    for i in prange(n):
        # 1. Accumulate inflow from subnetwork and upstream
        total_inflow = subnetwork_outflow[i] + upstream_inflow[i]

        # 2. Kinematic wave routing
        outflow = kinematic_wave_routing(
            storage=channel_storage[i],
            inflow=total_inflow,
            slope=channel_slope[i],
            width=channel_width[i],
            manning=channel_manning[i],
            delta_t=delta_t
        )

        # 3. Update storage
        channel_storage[i] += (total_inflow - outflow) * delta_t

        # 4. Calculate depth and velocity
        depth = channel_storage[i] / (channel_width[i] * channel_length[i])
        cross_section = depth * channel_width[i]
        velocity = outflow / cross_section if cross_section > 0 else 0

        # 5. Output to downstream cell
        channel_outflow_downstream[downstream_id[i]] = outflow
```

#### 4. Kinematic Wave Solver (`main_channel/kinematic_wave.py`)

**Purpose:** Solve kinematic wave equation for channel flow

**Equation:**
```
∂Q/∂t + ∂Q/∂x = lateral_inflow
Q = α * A^β  (Manning's equation form)

where:
α = (1/n) * S^(1/2)
β = 5/3 (for rectangular channel)
```

**Algorithm (Newton-Raphson iteration):**
```python
def kinematic_wave_routing(storage, inflow, slope, width, manning, delta_t):
    # Initial guess
    Q_out = inflow

    # Newton-Raphson iteration (converge to solution)
    for iteration in range(max_iterations):
        # Storage-discharge relationship
        A = (Q_out * manning / (width * slope^0.5))^0.6
        S = A * channel_length

        # Residual: F = S_new - S_old - (Q_in - Q_out) * dt
        residual = S - storage - (inflow - Q_out) * delta_t

        if abs(residual) < tolerance:
            break

        # Derivative: dF/dQ
        dF_dQ = dS_dQ - delta_t

        # Update: Q_new = Q_old - F / (dF/dQ)
        Q_out -= residual / dF_dQ

    return Q_out
```

#### 5. Reservoir Regulation (`reservoirs/regulation.py`)

**Purpose:** Update reservoir storage and release water

**Algorithm:**
```python
def regulation(reservoir_storage, channel_outflow, reservoir_release, ...):
    for i in prange(n_reservoirs):
        grid_idx = reservoir_grid_index[i]

        # 1. Calculate inflow
        inflow = channel_outflow[grid_idx] * delta_t

        # 2. Calculate outflow
        release_rate = reservoir_release[i]
        outflow = release_rate * delta_t

        # 3. Calculate evaporation
        evaporation = potential_evap * surface_area[i] * delta_t

        # 4. Update storage
        new_storage = reservoir_storage[i] + inflow - outflow - evaporation

        # 5. Apply constraints
        if new_storage > storage_capacity[i]:
            # Exceed capacity: increase release to spill
            outflow = inflow + reservoir_storage[i] - storage_capacity[i] - evaporation
            new_storage = storage_capacity[i]
        elif new_storage < min_storage:
            # Below minimum: reduce release
            outflow = max(0, reservoir_storage[i] + inflow - min_storage - evaporation)
            new_storage = min_storage

        reservoir_storage[i] = new_storage

        # 6. Route outflow downstream
        channel_outflow[downstream_id[grid_idx]] += outflow / delta_t
```

#### 6. Reservoir Release Calculation (`reservoirs/release.py`)

**Purpose:** Determine monthly reservoir release targets

**Biemans et al. (2011) Generic Rules:**
```python
def regulation_release(grid, state, config):
    for reservoir in reservoirs:
        # Expected inflow and demand for the month
        expected_inflow = reservoir_streamflow_schedule[reservoir, month]
        expected_demand = reservoir_demand_schedule[reservoir, month]

        # Pre-release (base release before adjustment)
        if has_irrigation_use:
            prerelease = expected_demand
        else:
            prerelease = expected_inflow

        # Storage adjustment factor
        storage_start_year = reservoir_storage_at_year_start[reservoir]
        k = storage_start_year / (regulation_param * storage_capacity)
        k = clip(k, min=0.5, max=1.5)  # Limit adjustment

        # Final release
        release = k * prerelease
        release = clip(release, min=min_release, max=max_release)

        reservoir_release[reservoir] = release
```

**ISTARF Data-Driven Rules (if enabled):**
```python
def istarf_release(grid, state, config):
    # Use reservoir-specific coefficients from database
    # Based on observed operational patterns
    for reservoir in reservoirs:
        if has_istarf_data:
            # Apply data-driven release function
            release = istarf_function(
                storage=reservoir_storage[reservoir],
                inflow=reservoir_inflow[reservoir],
                month=current_month,
                coefficients=istarf_coefficients[reservoir]
            )
            reservoir_release[reservoir] = release
```

---

## Data Flow

### Input Data Flow

```
NetCDF Input Files
    ↓
┌─────────────────────┐
│   Grid File         │
├─────────────────────┤
│ - Domain definition │
│ - River network     │
│ - Parameters        │
│ - Reservoir data    │
└─────────────────────┘
    ↓
  Grid Object
    ↓
┌─────────────────────┐  ┌─────────────────────┐
│  Runoff Files       │  │   Demand Files      │
├─────────────────────┤  ├─────────────────────┤
│ - Surface runoff    │  │ - Water demand      │
│ - Subsurface runoff │  │   (or Farmer ABM    │
│ - Wetland runoff    │  │    calculation)     │
└─────────────────────┘  └─────────────────────┘
    ↓                        ↓
  State Object (at each timestep)
    ↓
┌─────────────────────────────────────┐
│      Routing Algorithms             │
├─────────────────────────────────────┤
│ Hillslope → Subnetwork → Channel    │
│         ↓                           │
│    Reservoir Regulation             │
│         ↓                           │
│   Water Supply Extraction           │
└─────────────────────────────────────┘
    ↓
  Updated State
    ↓
┌─────────────────────┐
│   Output Buffer     │
├─────────────────────┤
│ - Accumulate values │
│ - Average over      │
│   output period     │
└─────────────────────┘
    ↓
NetCDF Output Files
```

### State Update Flow

```
Time: t
State: S(t)
    ↓
┌──────────────────┐
│ Read Forcing     │
│ - Runoff         │
│ - Demand         │
└──────────────────┘
    ↓
┌──────────────────┐
│ Routing Loop     │
│ (subcycles)      │
│                  │
│ FOR i = 1 to N:  │
│   Hillslope      │
│   Subnetwork     │
│   Main Channel   │
│   Reservoirs     │
└──────────────────┘
    ↓
State: S(t + Δt)
    ↓
┌──────────────────┐
│ Write Output     │
│ (if period end)  │
└──────────────────┘
```

---

## Entry Points

### 1. Command-line Tools

```bash
# Download sample data
python -m mosartwmpy.download

# Create reservoir parameters from GRanD database
create_grand_parameters

# Convert elevation data to Parquet
bil_to_parquet
```

### 2. Standalone Python Script

```python
from mosartwmpy import Model

# Create model instance
model = Model()

# Initialize from config file
model.initialize('./config.yaml')

# Run simulation
model.update_until(model.get_end_time())

# Cleanup
model.finalize()
```

### 3. BMI Coupled Mode (e.g., with CLM)

```python
from mosartwmpy import Model

# Initialize
model = Model()
model.initialize('./config.yaml')

# Coupling loop
for t in range(num_timesteps):
    # Get runoff from land model
    surface_runoff = clm.get_output('surface_runoff')
    subsurface_runoff = clm.get_output('subsurface_runoff')

    # Set as mosartwmpy input
    model.set_value('surface_runoff_flux', surface_runoff)
    model.set_value('subsurface_runoff_flux', subsurface_runoff)

    # Advance mosartwmpy
    model.update()

    # Get river discharge back to CLM (if needed)
    discharge = model.get_value('outgoing_water_volume_transport_along_river_channel')
    clm.set_input('river_discharge', discharge)

model.finalize()
```

### 4. Interactive Visualization

```python
from mosartwmpy import Model
from mosartwmpy.plotting.plot import plot_variable, plot_reservoir

# Run simulation
model = Model()
model.initialize('./config.yaml')
model.update_until(model.get_end_time())

# Plot spatial variable
model.plot_variable('outgoing_water_volume_transport_along_river_channel',
                    log_scale=True)

# Plot time series at specific location
plot_variable(model=model,
              variable='RIVER_DISCHARGE_OVER_LAND_LIQ',
              latitude=40.5, longitude=-105.5,
              start='1981-05-01', end='1981-05-31')

# Plot reservoir operation
plot_reservoir(model=model,
               grand_id=310,  # GRanD database ID
               start='1981-05-01', end='1981-05-31')
```

---

## Key Dependencies

### Core Scientific Stack
- **numpy** (1.20-1.99) - Array operations
- **xarray** (0.19+) - Gridded data I/O
- **pandas** (1.3+) - Time series and DataFrames
- **netCDF4** (1.5.7+) - NetCDF file I/O

### Performance
- **numba** (0.53+) - **Critical**: JIT compilation for routing algorithms
- **dask** (2021.10+) - Parallel/distributed computing

### Geospatial
- **geopandas** (0.10+) - Geospatial operations
- **rioxarray** (0.8+) - Raster I/O
- **pyarrow** (6.0+) - Parquet file I/O

### Model Framework
- **bmipy** (2.0+) - Basic Model Interface
- **benedict** (0.24+) - YAML configuration

### Optimization (optional)
- **pyomo** (6.2+) - For farmer ABM optimization

### Visualization (optional)
- **matplotlib** (3.4+) - 2D plotting
- **hvplot** (0.7+) - Interactive plots
- **contextily** (1.2+) - Basemap tiles

---

## Performance Considerations

### Numba JIT Compilation

**All core routing functions use Numba:**
- `hillslope_routing()`
- `subnetwork_routing()`
- `main_channel_routing()`
- `kinematic_wave_routing()`
- `regulation()`
- `extraction_regulated_flow()`

**Benefits:**
- 10-100x speedup over pure Python
- Parallel execution with `prange()`
- Compiled on first call (slight startup delay)

### Domain Masking

**Active cell masking:**
- Trims arrays to only active cells (land + routing)
- Reduces memory usage
- Improves cache locality

**Unmask for output:**
- `unmask_output=true` in config
- Restores full grid for visualization

### Subcycling

**Purpose:** Numerical stability for fast-flowing cells

**Configuration:**
```yaml
simulation:
  subcycles: 10           # Subdivide timestep (default: 10)
  routing_iterations: 1   # Iterations per subcycle (default: 1)
```

**Effect:**
- Effective timestep = timestep / subcycles
- Trade-off: accuracy vs. computation time

### Vectorization

**Spatial parallelism:**
- All grid cells computed independently
- Numba parallelizes with `prange(n_cells)`
- Efficient on multi-core systems

**Temporal serialization:**
- Timesteps must be sequential
- Upstream/downstream dependencies

---

## Testing and Validation

### Unit Tests (`tests/`)

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_model.py::test_model_initialize
```

**Test Coverage:**
- Model initialization
- BMI interface compliance
- Routing functions
- Reservoir operations
- I/O operations

### Validation Framework (`validation/`)

**Compare against reference:**
- Load reference simulation output
- Run test simulation
- Calculate NMAE (Normalized Mean Absolute Error)
- Check spatial sums for key variables

**Usage:**
```bash
cd validation
python validation.py
```

---

## Common Workflows

### 1. Quick Start (Sample Data)

```bash
# Download sample data
python -m mosartwmpy.download

# Run sample simulation
python launch.py

# Output: ./output/test/test_1981_05.nc
```

### 2. Custom Domain Simulation

```yaml
# config.yaml
grid:
  path: "./input/domains/my_domain.nc"
  subdomain:
    - [35.0, -120.0]  # [lat, lon] corners
    - [45.0, -110.0]

runoff:
  path: "./input/runoff/my_runoff-*.nc"

simulation:
  start_date: "2000-01-01"
  end_date: "2010-12-31"
  output: [
    "RIVER_DISCHARGE_OVER_LAND_LIQ",
    "STORAGE_LIQ",
    "WRM_SUPPLY"
  ]
```

```bash
python launch.py
```

### 3. Restart Simulation

```yaml
# config.yaml
simulation:
  restart_file: "./output/test/restart_1981.nc"
  start_date: "1981-01-01"  # Will start from restart time
  end_date: "1982-12-31"
```

### 4. Visualization Workflow

```python
from mosartwmpy import Model
from mosartwmpy.plotting.plot import plot_variable

# Load completed simulation
model = Model()
model.initialize('./config.yaml')
model.update_until(model.get_end_time())

# Spatial plot (final state)
model.plot_variable('channel_storage', log_scale=True)

# Time series at location
plot_variable(model=model,
              variable='RIVER_DISCHARGE_OVER_LAND_LIQ',
              latitude=40.0, longitude=-105.0,
              start='1981-01-01', end='1981-12-31')
```

---

## Additional Resources

### Documentation
- **Sphinx Docs:** `docs/` directory
- **Tutorial Notebook:** `notebooks/tutorial.ipynb`
- **JOSS Paper:** `paper/paper.md`

### Key Papers
- Biemans et al. (2011) - Reservoir regulation rules
- Li et al. (2013) - MOSART model description
- Thurber et al. (2021) - mosartwmpy implementation (JOSS)

### Repository
- **GitHub:** https://github.com/IMMM-SFA/mosartwmpy
- **Issues:** Report bugs and request features
- **Contributing:** See CONTRIBUTING.md

---

## Summary

**mosartwmpy** is a sophisticated water routing and management model with:

1. **Multi-scale routing:** Hillslope → Subnetwork → Main Channel
2. **Water management:** Reservoir operations and demand satisfaction
3. **BMI interface:** Couples with Earth system models
4. **Performance:** Numba-accelerated core algorithms
5. **Flexibility:** Configurable via YAML, supports various workflows
6. **Open source:** Well-documented, tested, peer-reviewed

**Key Design Patterns:**
- **Separation of concerns:** Grid, State, Config, I/O, Routing
- **Numba acceleration:** JIT-compile performance-critical loops
- **BMI standard:** Interoperable with other models
- **Configuration-driven:** Easy to customize without code changes
- **Modular:** Easy to extend (e.g., add new reservoir rules)

This architecture enables efficient simulation of continental-scale river networks with integrated water management.
