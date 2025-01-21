![Logo](https://raw.githubusercontent.com/deober/phaselocker/f16ed50974a8bac16e6201ef7cdf5cfb368dbcf6/phaselocker_logo_with_text.svg?token=AMM2ZZUYVKP3BIWGFTOTASLHSAHZU)

A minimal python package for Bayesian cluster expansion. Optionally for finding, enforcing and sampling cluster expansion ground state cones.

For context, see https://doi.org/10.1103/PhysRevMaterials.8.103803 

Requires numpy, scipy and bokeh. Example notebooks may use additional packages. 

### Modules:  

#### phaselocker.geometry:  
  Contains most functions; everything necessary for quantifying ground state accuracy, finding ground state cones, finding lower convex hulls, etc.  
  
#### phaselocker.sampling:  
  Contains a basic Monte Carlo sampler and some basic ECI prior currying functions.  
  
#### phaselocker.visualization:  
  Contains a bokeh plotting function to visualize formation energies and lower convex hulls in binary alloys, as well as as a box and whiskers plot to visualize ECI posterior uncertainties. 
