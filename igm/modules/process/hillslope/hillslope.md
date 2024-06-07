
### <h1 align="center" id="title">IGM module `hillslope_erosion` </h1>

# Description:

This IGM module implements changes in topography due to diffusive hillslope processes (e.g. rainsplash, soil creep, bioturbation). 
The topography is updated (with a frequency provided by parameter `hillslope_update_freq`) using a nonlinear diffusion equation after Roering et al. (1999) and Pelletier (2008). 
The practical implementation follows David Egholm's iSOSIA code (https://github.com/davidlundbek/iSOSIA), translated into python/tensorflow.
Hillslope diffusion counteracts the formation of unrealistically steep slopes that result from glacial erosion.

Roering, J. J., Kirchner, J. W. & Dietrich, W. E. Evidence for nonlinear, diffusive
sediment transport and implications for landscape morphology. Wat. Resour. Res.
35, 853–870 (1999).

Pelletier, J. Quantitative Modeling of Earth Surface Processes (Cambridge Univ.
Press, 2008).