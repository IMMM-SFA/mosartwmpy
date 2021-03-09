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
Though often overlooked, the developer experience can have a large downstream influence on the quality, legibility, and extendability of a codebase. Scientists and data wranglers alike have come to expect the ease and portability of working with frameworks like `numpy` and `pandas`, while dreading the need to fiddle with the intricacies of compilers and interpreters on instituitonal computing systems. This dread can often translate to hours lost in scanning cryptic log messages, tracking down an esoteric version of a system library, or just silently seething at the scientists that came before for their arcane and indiscernable code. `mosartwmpy` is a Python reimplementation of `MOSART-WM` [@voisin2013improved] -- a hydrological model for water routing and reservoir management written in Fortran -- that focuses on the developer experience by being intuitive, lightweight, well-documented, and interoperable.

![River basin flow as output from `mosartwmpy`.](figure_1.png){ width=50% }

# Summary
MOSART (MOdel for Scale Apaptive River Transport) was developed as a scalable framework for representing and studying riverine dynamics of water, energy, and biogeochemistry cycles across local, regional and global scales from an integrated human-earth system perspective [@li2013physically]. In particular, a water management module for dams/reservoirs has been embedded within MOSART (denoted as MOSART-WM) to simulate the effects of dam operations as well as surface-water withdrawal on downstream flow and water temperature across a range of spatial scales [@voisin2013improved].

@voisin2013improved introduced MOSART-WM as a large scale spatially distributed reservoir and water management model which is coupled with the river routing MOSART model and represents the spatially distributed withdrawals, as well as withdrawals from river channels and diversion from reservoir releases. River regulation is represented through reservoirs (up to one per grid cell). Each reservoir release is independently calibrated using long term mean monthly inflow into the reservoir, long term monthly demand associated with this reservoir, and reservoir goals (flood control, irrigation, recreation, etc.). Generic monthly pre-release rules and storage targets are set up for individual reservoirs however those releases are updated annually for inter-annual variability (dry or wet year) and for daily constraints such as environmental flow minimun release, spill and minimum storage. The WM model allows evaluating the impact of water management over multiple river basins at once (global, continental scales) and with consistent representation of human operations over the full domain.

Today, MOSART-WM is often utilised as the hydrological component in a larger suite of earth-science models, such as in @doecode_10475. In this incarnation, MOSART-WM is quite efficient and streamlined when running on a supported HPC cluster. But, learning to use this complex codebase can take quite a while, especially with the common need to wait in a queue in order to run the code on a supported machine. This slow turnaround time further complicates the ability to add new features to the existing codebase. Imagine waiting half a day or even just half an hour between writing new code and being able to test it. Furthermore, early career scientists and engineers rarely know Fortran or C, or the intricacies of working with the various compilers and architectures.

Enter `mosartwmpy`, a refactor of MOSART-WM in Python which aims to eliminate the noise by being easy to learn, use, and modify. Python has the advantages of being widely known by early career scientists and engineers, and of being able to run on nearly any platform without much fuss. `mosartwmpy` implements the Community Surface Dynamics Modeling System (CSDMS) Basic Model Interface (BMI) [@peckham2013component] [@hutton2020basic], which offers a familiar set of controls for operating and coupling the model. `mosartwmpy` can be accessed on GitHub (https://github.com/IMMM-SFA/mosartwmpy), and a walkthrough of key functionality and use can be found here [Tutorial](https://mosartwmpy.readthedocs.io/en/latest/). Model results have been validated against historical simulations [@https://doi.org/10.1029/2020WR027902], and a validation utility is included with the code.

Two new model features (and associated scientific papers) are already under development in the new paradigm: an Agent Based Model (ABM) for more accurately simulating irrigation demand based on the economics of various crop types led by Jim Yoon, and an improved reservoir operations module for more accurately simulating reservoir release based on data-driven harmonic functions led by Sean Turner. The turnaround time for implementing these new features in the codebase is proving to be much quicker than past work.

# Acknowledgements
This research was supported in part by the U.S. Department of Energy, Office of Science, as part of research in MultiSector Dynamics, Earth and Environmental System Modeling Program. The Pacific Northwest National Laboratory is operated for DOE by Battelle Memorial Institute under contract DE-AC05-76RL01830. The views and opinions expressed in this paper are those of the authors alone.

# References
