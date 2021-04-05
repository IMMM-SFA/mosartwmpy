---
title: 'mosartwmpy: A Python implementation of the MOSART-WM coupled hydrologic routing and water management model'
tags:
  - Python
  - hydrology
  - water management
  - multisector dynamics
  - reservoir modeling
authors:
  - name: Travis Thurber
    orcid: 0000-0002-4370-9971
    affiliation: 1
  - name: Chris R. Vernon
    orcid: 0000-0002-3406-6214
    affiliation: 1
  - name: Ning Sun
    orcid: 0000-0002-4094-4482
    affiliation: 1
  - name: Sean W. D. Turner
    orcid: 0000-0003-4400-9800
    affiliation: 1
  - name: Jim Yoon
    orcid: 0000-0002-8025-2587
    affiliation: 1
  - name: Nathalie Voisin
    orcid: 0000-0002-6848-449X
    affiliation: 1
affiliations:
 - name: Pacific Northwest National Laboratory, Richland, WA., USA
   index: 1
date: 22 March 2021
bibliography: paper.bib
---

# Summary
`mosartwmpy` is a python implementation of the Model for Scale Adaptive River Transport with Water Management (MOSART-WM). This new version retains the functionality of the legacy model (written in FORTRAN) while providing new features to enhance user experience and extensibility. MOSART is a large-scale river-routing model used to study riverine dynamics of water, energy, and biogeochemistry cycles across local, regional, and global scales [@li2013physically]. The WM component introduced by @voisin2013improved represents river regulation through reservoir storage and release operations, diversions from reservoir releases, and allocation to sectoral water demands. Each reservoir release is independently calibrated using long-term mean monthly inflow into the reservoir, long-term mean monthly demand associated with this reservoir, and reservoir goals (flood control, irrigation, recreation, etc.). Generic monthly pre-release rules and storage targets are set up for individual reservoirs; however, those releases are updated annually for inter-annual variability (dry or wet year) and daily for environmental constraints such as flow minimum release and minimum/maximum storage levels. The WM model allows an evaluation of the impact of water management over multiple river basins at once (global, continental scales) and with consistent representation of human operations over the full domain.

# Statement of Need
MOSART-WM is often utilized as the hydrological component in a larger suite of earth-science models, such as in @doecode_10475. In this context, MOSART-WM is quite efficient and streamlined when running on a supported High-Performance Computing (HPC) cluster. However, learning how to use, extend, and test a tightly-coupled codebase written with domain knowledge implied in a lower-level programming language may greatly increase the turnaround time and error rate for setting up and executing novel experiments. Broadening the code’s accessibility using a programming language such as Python, which in 2020 was the second most utilized language on GitHub [@octoverse_2020], provides a more accessible option to learn and contribute to the science on most computational platforms.

`mosartwmpy` was designed to bridge the gap between the domain scientist who wants a performant software that can still be extended for future research needs, and the new user who may not have expertise within the hydrologic sciences but wishes to integrate the process into their own workflow or quickly become capable in conducting hands-on experimentation for educational purposes. A refactor of MOSART-WM in Python ameliorates the steep learning curve of the FORTRAN version by providing an easy to learn, use, and modify interface. `mosartwmpy` was also built with interoperability in mind by implementing the Community Surface Dynamics Modeling System (CSDMS) Basic Model Interface (BMI) [@peckham2013component; @hutton2020basic], which offers a familiar set of controls for operating and coupling the model. `mosartwmpy` can be accessed on GitHub (https://github.com/IMMM-SFA/mosartwmpy), and a walkthrough of key functionality and use can be found here: [tutorial](https://mosartwmpy.readthedocs.io/en/latest/). Model results have been validated against historical simulations [@https://doi.org/10.1029/2020WR027902], and a validation utility is included with the code.

# Functionality and limitations
`mosartwmpy` replicates the entirety of the physics found in MOSART-WM. Input consists of spatially distributed surface and subsurface runoff (i.e. excess water on and below the ground that can move on a 2D plane as driven by gravity). The hillslope and subsurface runoff populate the local streams and river channels, which generally flow toward the oceans (as in Figure \autoref{@fig:one}). In many locations along the main channels, reservoirs collect and store water, releasing portions of the storage downstream over time for various societal uses (i.e. hydroelectricity, drinking water, flood control, irrigation). Water is also consumed directly from the channels and tributaries for irrigation or non irrigation uses.

![River basin flow as output from `mosartwmpy`.\label{fig:one}](figure_1.png){ width=75% }

In the 1/8 degree continental United States (CONUS) configuration of `mosartwmpy` [@voisin2017conus], the hillslope routing is set to a default one-hour resolution. To ensure mass conservation in the routing process while allowing rivers to travel across multiple grid cell within one time step, sub time steps are applied for resolving the channel flow accounting for reservoir regulation and water withdrawals. The reservoir extraction is performed at a 20-minute resolution, and the channel flow at resolutions ranging from 12 minutes to half a minute based on the complexity of the grid cell.

Any spatial or simulated variable can be written to an output file in NetCDF format at predetermined intervals, typically averaged to daily values. The model can also save its current state as a compressed restart file, allowing the simulation to be regularly checkpointed.

Currently, a 1/8 degree CONUS simulation can efficiently scale across eight CPUs using C-bindings to parallelize the vectorized calculations; limitations in the Python vector math implementations restrict  performance boosts of additional processors. Further enhancement is required to match the speeds obtainable in the Fortran version of MOSART-WM, which can effectively scale across a greater number of CPUs by distributing cell-by-cell scalar calculations. As a sample benchmark, a one-year simulation using MOSART-WM on a 24-CPU node of an institutional supercomputer can be performed in about half an hour; whereas the same simulation using `mosartwmpy` on a personal 6-CPU laptop can be performed in about five hours.

# Ongoing Research
As open-source software, `mosartwmpy` promotes collaborative and community model development to meet scientific objectives. Two examples of the potential to extend this codebase are currently underway as a part of the U.S. DOE’s Department of Energy’s Integrated MultiSector MultiScale MultiModel (IM3) basic research project: an Agent Based Model (ABM) for more accurately simulating irrigation demand based on the economics of various crop types [@abm; @Yoone2020431118], and an improved reservoir operations module for more accurately simulating reservoir release based on data-driven harmonic functions [@reservoirs]. Future planned experiments will study the uncertainty characterization and quantification [@hess-19-3239-2015] of the model based on perturbations in key parameters.

# Acknowledgements
This research was supported in part by the U.S. Department of Energy, Office of Science, as part of research in MultiSector Dynamics, Earth and Environmental System Modeling Program. The Pacific Northwest National Laboratory is operated for DOE by Battelle Memorial Institute under contract DE-AC05-76RL01830. The views and opinions expressed in this paper are those of the authors alone.

# References
