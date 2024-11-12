import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import mconf.mconf as mconf
import numpy as np
import os 
os.chdir('J_0_test')
 
mconf_config = {'B0': 2.525,
                'B0_angle': 0.0,
                'accuracy': 1e-10, #accuracy of magnetic to cartesian coordinat transformation
                'truncation': 1e-10} #trancation of mn harmonics
eq = mconf.Mconf_equilibrium('w7x-sc1.bc',mconf_config=mconf_config)
point = [-0.10411962,  5.01378673,  0.19537387]
#point =  [-0.1073791,   5.00117022,  0.19425995]
s0, vecB = eq.get_B(point)
 
L = 20; N = 10000; 
from scipy.integrate import solve_ivp
rhs_B_fortran1 = lambda t, y:  eq.get_B(y)[1]
 
r1 = np.array(eq.mag2xyz(s0,0.,0.))
field_line_sol1 = solve_ivp(rhs_B_fortran1, [0,L], r1, method = 'RK45', max_step = L/N, atol=1., dense_output=True)
#field_line_sol1 = solve_ivp(rhs_B_fortran1, [0,-L], r1, method = 'RK45', max_step = L/N, atol=1., dense_output=True)
 
s,B = eq.get_s_B_T(field_line_sol1.y[0],field_line_sol1.y[1],field_line_sol1.y[2])
magB = np.linalg.norm(B,axis=1)
path = np.hstack([0,np.cumsum(np.sqrt(np.sum((np.diff(field_line_sol1.y, axis=-1)**2),axis=0)))])
 
#matplotlib widget
fig, ax = plt.subplots(figsize=(6,5))
 
ax.plot(path,magB)
from scipy.integrate import cumtrapz
#cumtrapz(np.sqrt(XYI - magB),x=path)
 
ax.grid()
plt.show()