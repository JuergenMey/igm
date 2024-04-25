
### <h1 align="center" id="title">IGM module `hillslope_erosion` </h1>

# Description:

This IGM module implements changes in topography due to diffusive hillslope processes (e.g. rainsplash, soil creep, bioturbation). The topography is updated (with a frequency provided by parameter `hillslope_diffusion_update_freq`) using an implicit scheme to solve the linear diffusion equation for a DEM using a default hillslope diffusivity of 1 m^2/yr (parameter `hillslope_diffusivity`). 
Hillslope diffusion counteracts the formation of unrealistically steep slopes that result from glacial erosion.