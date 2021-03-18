---
title: 'mosartwmpy: A Python implementaion of the MOSART-WM coupled hydrologic routing and water management model'
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
affiliations:
 - name: Pacific Northwest National Laboratory, Richland, WA., USA
   index: 1
date: 21 March 2021
bibliography: paper.bib
---

# Statement of need
Developer experience [@devex] can have a large downstream influence on the quality, usability, and extensibility of a codebase [@prechelt; @nanz]. Scientists and data wranglers alike have come to expect the ease and portability of working with libraries such as NumPy and Pandas, without the need to struggle with the intricacies of compilers and interpreters on institutional computing systems. This added effort can often translate to hours lost in scanning cryptic log messages or tracking down an esoteric version of a system library when hoping to add or improve a software feature. `mosartwmpy` is a Python reimplementation of MOSART-WM [@voisin2013improved] -- a hydrological model for water routing and reservoir management written in Fortran -- that focuses on the developer experience by being intuitive, lightweight, well-documented, extensible, and interoperable.

![River basin flow as output from `mosartwmpy`.\label{fig:1}](figure_1.png){ width=50% }

# Summary
MOSART (MOdel for Scale Apaptive River Transport) was developed as a scalable framework for representing and studying riverine dynamics of water, energy, and biogeochemistry cycles across local, regional and global scales from an integrated human-earth system perspective [@li2013physically]. Additionally, @voisin2013improved introduced a component to MOSART named WM (Water Management) as a large scale spatially distributed reservoir and water management model which is tightly-coupled with the river routing MOSART model and represents the spatially distributed withdrawals, as well as withdrawals from river channels and diversion from reservoir releases. River regulation is represented through reservoirs (up to one per grid cell). Each reservoir release is independently calibrated using long term mean monthly inflow into the reservoir, long term monthly demand associated with this reservoir, and reservoir goals (flood control, irrigation, recreation, etc.). Generic monthly pre-release rules and storage targets are set up for individual reservoirs however those releases are updated annually for inter-annual variability (dry or wet year) and for daily constraints such as environmental flow minimum release, spill and minimum storage. The WM model allows an evaluation of the impact of water management over multiple river basins at once (global, continental scales) and with consistent representation of human operations over the full domain. 

MOSART-WM is often utilized as the hydrological component in a larger suite of earth-science models, such as in @doecode_10475. In this context, MOSART-WM is quite efficient and streamlined when running on a supported High-Performance Computing (HPC) cluster. However, learning how to use, extend, and test a tightly-coupled codebase written with domain knowledge implied in a lower-level programming language may greatly reduce the userbase. Broadening the code’s accessibility using a programming language such as Python, which in 2020 was the second most utilized language on GitHub [@octoverse_2020], provides a more accessible option to learn and contribute to the science on most computational platforms.

`mosartwmpy` was designed to bridge the gap between domain scientist who want a performant software that can still be extended for future research needs and the new user who may not have expertise within the hydrologic sciences but wishes to either 1) integrate the process into their own workflow, or 2) wants to quickly become capable in conducting hands-on experimentation for educational purposes. A refactor of MOSART-WM in Python ameliorates the steep learning curve of the FORTRAN version by providing an easy to learn, use, and modify interface. `mosartwmpy` was also built with interoperability in mind by implementing the Community Surface Dynamics Modeling System (CSDMS) Basic Model Interface (BMI) [@peckham2013component; @hutton2020basic], which offers a familiar set of controls for operating and coupling the model. `mosartwmpy` can be accessed on GitHub (https://github.com/IMMM-SFA/mosartwmpy), and a walkthrough of key functionality and use can be found here: [tutorial](https://mosartwmpy.readthedocs.io/en/latest/). Model results have been validated against historical simulations [@https://doi.org/10.1029/2020WR027902], and a validation utility is included with the code.

As an open-source software, `mosartwmpy` promotes collaborative and community model development to meet scientific objectives. Two examples of the potential to extend this codebase are currently underway as a part of the U.S. DOE’s Department of Energy’s Integrated MultiSector MultiScale MultiModel (IM3) basic research project: an Agent Based Model (ABM) for more accurately simulating irrigation demand based on the economics of various crop types [@abm], and an improved reservoir operations module for more accurately simulating reservoir release based on data-driven harmonic functions [@reservoirs].

# Functionality and limitations
`mosartwmpy` replicates the entirety of the physics found in MOSART-WM. Input consists of surface, subsurface, and wetland runoff (i.e. rain and snow melt that flows downhill over or under the land). The flow of runoff populates the river channels and tributaries, which generally flow toward the oceans [@fig:1]. In many locations along the main channels, reservoirs collect and store water from the channels for various uses by humanity (i.e. hydroelectricity, drinking water), and release portions of the storage downstream over time. Water is also consumed directly from the channels and tributaries for irrigation or industry.

In `mosartwmpy`, these interactions are considered using multiple, dynamic timescales. On a 1/8 degree grid of the United States for instance, the overland flow is usually considered at a one-hour resolution, the reservoir extraction at a 20-minute resolution, and the channel flow at resolutions ranging from 12 minutes to half a minute based on the complexity of the grid cell.

Any spatial or simulated variable can be written to an output file in the NetCDF format at predetermined intervals, typically at a daily averaged resolution. Similarly, the model can write out the entirety of its current state as a compressed restart file, allowing the simulation to be regularly checkpointed.

Currently, the simulation can efficiently scale across ten or so CPUs by using C-bindings to parallelize the vectorized calculations. More work needs to be done in this area to measure up to the speeds obtainable in the Fortran version of MOSART-WM, which can effectively scale across any number of CPUs by distributing cell-by-cell calculations. For this reason, `mosartwmpy` has the advantage of being able to run on any system that supports Python, but the disadvantage of being significantly slower for large scale simulations. This disparity will be addressed in ongoing work.

# Acknowledgements
This research was supported in part by the U.S. Department of Energy, Office of Science, as part of research in MultiSector Dynamics, Earth and Environmental System Modeling Program. The Pacific Northwest National Laboratory is operated for DOE by Battelle Memorial Institute under contract DE-AC05-76RL01830. The views and opinions expressed in this paper are those of the authors alone.

# References
