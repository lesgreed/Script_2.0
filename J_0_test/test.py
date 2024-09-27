import mconf
import numpy as np
import os 
os.chdir('J_0_test')
print("Current working directory:", os.getcwd())
            
    
mconf_config = {'B0': 2.525, #B field at magentic axis at toroidal angle B0_angle
                  'B0_angle': 0.0,
                  'extraLCMS': 1.8,   #Flux surfaces extrapolation parameter (s_max)
                  'accuracy': 1e-10, #accuracy of magnetic to cartesian coordinat transformation
                  'truncation': 1e-10}  #truncation when generating grid 
eq0 = mconf.Mconf_equilibrium('w7x-sc1.bc', mconf_config)
