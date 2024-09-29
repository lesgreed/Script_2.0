import mconf
import numpy as np
import os 
os.chdir('J_0_test')
print("Current working directory:", os.getcwd())
            
try:
    mconf = np.ctypeslib.load_library('mconf_matlab', loader_path='.')
    print("DLL loaded successfully.")
except Exception as e:
    print(f"Error loading DLL: {e}")


mconf_config = {'B0': 2.525,
                'B0_angle': 0.0,
                'accuracy': 1e-10, #accuracy of magnetic to cartesian coordinat transformation
                'truncation': 1e-10} #trancation of mn harmonics
eq = mconf.Mconf_equilibrium('w7x-sc1.bc',mconf_config=mconf_config)
 
# Getting B-field
tmp = eq.get_s_and_B(5.9,0.,0)
print(tmp)