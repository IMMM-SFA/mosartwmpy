---
title: 'wolfgang:  A Python implementaion of the MOSART-WM coupled hydrologic routing and water management model'
tags:
  - Python
  - Water management
  - MultiSector Dynamics
authors:
  - name: Travis Thurber
    orcid: need your ORCID
    affiliation: 1
  - name: Chris R. Vernon
    orcid: 0000-0002-3406-6214
    affiliation: 1
affiliations:
 - name: Pacific Northwest National Laboratory, Richland, WA., USA
   index: 1
date: 20 January 2021
bibliography: paper.bib
---

# Statement of need
This would describe why you build the Python implementaion of MOSART-WM.

![This is just a space holder for an image that will represent your new Python package.](figure_1.png)

# Summary
I just took this content dirctly from several existing repos to prime the pump and demonstrate how references are included in text; it needs to be rewritten for continutity:  

MOSART was developed as a scalable framework for representing and studying riverine dynamics of water, energy, and biogeochemistry cycles across local, regional and global scales from an integrated human-earth system perspective [@li2013physically]. In particular, a dam/reservoir module has been embedded within MOSART (denoted as MOSART-WM) to simulate the effects of dam operations as well as surface-water withdrawal on downstream flow and water temperature across a range of spatial scales [@voisin2013improved].

@voisin2013improved introduced MOSART-WM as a large scale spatially distributed reservoir and water management model which is coupled with the river routing MOSART model and represents the spatially distributed withdrawals, as well as withdrawals from river channels and diversion from reservoir releases. River regulation is represented through reservoirs (up to one per grid cell). Each reservoir release is independently calibrated using long term mean monthly inflow into the reservoir, long term mon monthly demand associated with this reservoir and reservoir goals (flood control, irrigation, other, combination of flood control with irrigation). Generic monthly pre-release rules and storage targets are set up for individual reservoirs however those releases are updated annually for inter-annual variability (dry or wet year) and for daily constraints such as environmental flow minimun release, spill and minimum storage. The WM model allows evaluating the impact of water management over multiple river basins at once (global, continental scales) and with consistent representation of human operations over the full domain.

`wolfgang` was implemented to...  

`wolfgang` is offers the following features...

`wolfgang` can be accessed on GitHub (https://github.com/IMMM-SFA/wolfgang). We provide an walthrough of some key functionality in a step-by-step tutorial in our website here: [Tutorial](link to the readthedocs.io that will have the tutorial).

# Acknowledgements
This research was supported in part by the U.S. Department of Energy, Office of Science, as part of research in MultiSector Dynamics, Earth and Environmental System Modeling Program. The Pacific Northwest National Laboratory is operated for DOE by Battelle Memorial Institute under contract DE-AC05-76RL01830. The views and opinions expressed in this paper are those of the authors alone.

# References
