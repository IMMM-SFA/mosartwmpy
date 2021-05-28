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
`mosartwmpy` is a Python implementation of the Model for Scale Adaptive River Transport with Water Management (MOSART-WM). This new version retains the functionality of the legacy model (written in FORTRAN) while providing new features to enhance user experience and extensibility. MOSART is a large-scale river-routing model used to study riverine dynamics of water, energy, and biogeochemistry cycles across local, regional, and global scales [@li2013physically]. The WM component introduced by @voisin2013improved represents river regulation through reservoir storage and release operations, diversions from reservoir releases, and allocation to sectoral water demands. Each reservoir release is independently calibrated using long-term mean monthly inflow into the reservoir, long-term mean monthly demand associated with this reservoir, and reservoir goals (flood control, irrigation, recreation, etc.). Generic monthly pre-release rules and storage targets are set up for individual reservoirs; however, those releases are updated annually for inter-annual variability (dry or wet year) and daily for environmental constraints such as flow minimum release and minimum/maximum storage levels. The WM model allows an evaluation of the impact of water management over multiple river basins at once (global, continental scales) and with consistent representation of human operations over the full domain.

# Statement of Need
MOSART-WM is often utilized as the hydrological component in a larger suite of earth-science models, such as in @doecode_10475. In this context, MOSART-WM is quite efficient and streamlined when running on a supported High-Performance Computing (HPC) cluster. However, learning how to use, extend, and test a complex codebase written with domain knowledge implied in a lower-level programming language may greatly increase the turnaround time and error rate for setting up and executing novel experiments. Broadening the code’s accessibility using a programming language such as Python, which in 2020 was the second most utilized language on GitHub [@octoverse_2020], provides a more accessible option to learn and contribute to the science on most computational platforms.

`mosartwmpy` was designed to bridge the gap between the domain scientist who wants a performant software that can still be extended for future research needs, and the new user who may not have expertise within the hydrologic sciences but wishes to integrate the process into their own workflow or quickly become capable in conducting hands-on experimentation for educational purposes. A refactor of MOSART-WM in Python ameliorates the steep learning curve of the FORTRAN version by providing an easy to learn, use, and modify interface. `mosartwmpy` was also built with interoperability in mind by implementing the Community Surface Dynamics Modeling System (CSDMS) Basic Model Interface (BMI) [@peckham2013component; @hutton2020basic], which offers a familiar set of controls for operating the model. By leveraging the BMI, `mosartwmpy` can be readily coupled with other earth system models to perform cross-domain experiments.

The target audience for `mosartwmpy` is the data scientist or hydrologist who wishes to rapidly prototype, test, and develop novel modeling capabilities relating to reservoirs and water demand. `mosartwmpy` can be accessed on GitHub (https://github.com/IMMM-SFA/mosartwmpy), and a walkthrough of key functionality and use can be found here: [tutorial](https://mosartwmpy.readthedocs.io/en/latest/). Model results have been validated against historical simulations [@https://doi.org/10.1029/2020WR027902], and a validation utility is included with the code.

# State of the field
In addition to `mosartwmpy`'s ancestor MOSART-WM, several other models are commonly used in hydrologic modeling, each excelling at different processes. The Community Land Model [CLM; @https://doi.org/10.1029/2018MS001583] focuses on the traditional water cycle (i.e. precipitation, evaporation) as well as plant and soil chemistry; runoff output from CLM can be used as input for `mosartwmpy`. StateMod [@statemod] focuses on water allocation based on legal constraints (water rights) as well as supply and demand. MODFLOW [@modflow] focuses on solving for complex groundwater flow properties on three-dimensional grids. GLOFRIM [@hoch2017glofrim] focuses on providing a framework to couple models implementing the BMI across multi-scale grids, for instance by coupling a meteorological model to a water cycle model to river routing model to hydrodynamic model. In this context, `mosartwmpy` focuses on the interactions between river routing, reservoir management, and water demand.

# Functionality
Model input for `mosartwmpy` consists of channel and reservoir geometry, groundwater and subsurface runoff (i.e. rain and ice melt), and water demand. The runoff populates the local streams and river channels, which generally flow toward the oceans (as in \autoref{fig:flow}). In many locations along the main channels, reservoirs collect and store water, releasing portions of the storage downstream over time for various societal uses (i.e. hydroelectricity, drinking water, flood control, and irrigation). Water is also consumed directly from the channels and tributaries for irrigation or non-irrigation uses.

![River basin flow over the continental United States as output from `mosartwmpy`.\label{fig:flow}](figure_1.png){ width=75% }

# Ongoing Research
As open-source software, `mosartwmpy` promotes collaborative and community model development to meet scientific objectives. Two examples of the potential to extend this codebase are currently underway as a part of the U.S. DOE’s Department of Energy’s Integrated Multisector Multiscale Modeling (IM3) basic research project: an Agent Based Model (ABM) for more accurately simulating irrigation demand based on the economics of various crop types [@abm; @Yoone2020431118], and an improved reservoir operations module for more accurately simulating reservoir release based on data-driven harmonic functions [@reservoirs]. Future planned experiments will study the uncertainty characterization and quantification [@hess-19-3239-2015] of the model based on perturbations in key parameters.

# Acknowledgements
This research was supported by the U.S. Department of Energy, Office of Science, as part of research in MultiSector Dynamics, Earth and Environmental System Modeling Program. The Pacific Northwest National Laboratory is operated for DOE by Battelle Memorial Institute under contract DE-AC05-76RL01830. The views and opinions expressed in this paper are those of the authors alone.

# References
