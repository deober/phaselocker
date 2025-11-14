![Logo](https://github.com/deober/phaselocker/blob/main/phaselocker_logo_with_text.svg)

A minimal python package for Bayesian cluster expansion. Optionally for finding, enforcing and sampling cluster expansion ground state cones. For context, see https://doi.org/10.1103/PhysRevMaterials.8.103803 and https://doi.org/10.1016/j.commatsci.2025.114330. The development of this software was supported by the National Science Foundation, under contract No. 2311370.


Requires numpy, scipy and bokeh. Example notebooks may use additional packages. 

### Modules:  

#### phaselocker.geometry:  
  Contains most functions; everything necessary for quantifying ground state accuracy, finding ground state cones, finding lower convex hulls, etc.  
  
#### phaselocker.sampling:  
  Contains a basic Monte Carlo sampler and some basic ECI prior currying functions.  
  
#### phaselocker.visualization:  
  Contains a bokeh plotting function to visualize formation energies and lower convex hulls in binary alloys, as well as as a box and whiskers plot to visualize ECI posterior uncertainties. 

#### phaselocker.fitting:
  Contains basic statistical learning fitting algorithms and model comparison tools.
