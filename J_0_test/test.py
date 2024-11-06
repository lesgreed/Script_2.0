import J_0_test.mconf.mconf as mconf
import numpy as np
import os 
os.chdir('J_0_test')
mconf_config = {'B0': 2.525,
                'B0_angle': 0.0,
                'accuracy': 1e-10, #accuracy of magnetic to cartesian coordinat transformation
                'truncation': 1e-10} #trancation of mn harmonics
eq = mconf.Mconf_equilibrium('J_0_test\w7x-sc1.bc',mconf_config=mconf_config)


def get_B(point):
 k, s = eq.get_B(point)
 return np.sqrt(s[0]**2+s[1]**2+s[2]**2)
def get_Bmax(point):
  B = eq.get_Bmax(point)
  return B
 




#print("Current working directory:", os.getcwd())
#point = np.array([306.12616, 456.41782, -36.95408])/100
#B = get_B(point)
#print("Current working directory:", os.getcwd())
#print(B)