# Phaselocker
![Logo](https://raw.githubusercontent.com/deober/phaselocker/762abd047e682e52978e9b8fdefd4415138a0663/phaselocker_logo.svg?token=AMM2ZZT2VWPPPYI277SIPHTHR7YCY)

A minimal necessary python package for finding, enforcing and sampling cluster expansion ground state cones.

For context, see https://doi.org/10.1103/PhysRevMaterials.8.103803 

Requires numpy, scipy, and bokeh. 

### Modules:  

#### phaselocker.geometry:  
  Contains most functions; everything necessary for quantifying ground state accuracy, finding ground state cones, finding lower convex hulls, etc.  
  
#### phaselocker.sampling:  
  Contains a basic Monte Carlo sampler and some basic ECI prior currying functions.  
  
#### phaselocker.visualization:  
  Contains a bokeh plotting function to visualize formation energies and lower convex hulls in binary alloys, as well as as a box and whiskers plot to visualize ECI posterior uncertainties. 
