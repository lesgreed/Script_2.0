import matplotlib.pyplot as plt
import numpy as np
import Surface_data as FuD
import NBI_Ports_data_input as Cout
import time
from matplotlib.path import Path
import os 
import J_0_test.mconf.mconf as mconf
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid as cumtrapz
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def transform(R, Z, phi):
    x = R   # Use R directly for x-axis
    y = Z  # Use Z directly for y-axis
    return x, y

def inverse_transform(R, Z, Phi):
    # Convert R, Z, and Phi back to x, y, z
    phi_radian = np.radians(Phi)
    x = R * np.cos(phi_radian)
    y = R * np.sin(phi_radian)
    z = Z
    return x, y, z
def trapezoidal_integral(f, s):
      ds = np.abs(np.diff(s))
      avg_f = (f[:-1] + f[1:]) / 2
      segment_integrals = ds * avg_f
      integral = np.sum(segment_integrals)
      return integral

def J_0_calculate(points,config, B):
     points = np.array(points, dtype=np.float64)
     previous_directory = os.getcwd()
     os.chdir('J_0_test')
    
     with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(calculate_J_0_for_point, points, [config] * len(points), [B] * len(points)), total=len(points)))
     print(config)
     os.chdir(previous_directory)
     return results

def calculate_J_0_for_point(point, config, B):
      #data type
      point = np.array(point, dtype=np.float64)

      #config
      mconf_config = {'B0': 2.525,
                'B0_angle': 0.0,
                'accuracy': 1e-10, #accuracy of magnetic to cartesian coordinat transformation
                'truncation': 1e-10} #trancation of mn harmonics
      eq = mconf.Mconf_equilibrium(config,mconf_config=mconf_config)
      

      #constant
      s0, vecB = eq.get_B(point)
      B_value = np.linalg.norm(vecB)
      B_max_point = eq.get_Bmax(s0)
      L = 300 
      N = 2000
      E_value = 50 * (1.6 * 10**(-19)) * 10**3  
      mu_value =  E_value/B

      #Solve eq
      def solve_differential(point, L, N, rhs_B):
        sol = solve_ivp(rhs_B, [0, L], point,method='RK45', max_step=L / N,atol=1e-6, dense_output=True)
        s_sol, B_sol = eq.get_s_B_T(sol.y[0],sol.y[1], sol.y[2])
        magB = np.linalg.norm(B_sol, axis=1)
        path = np.zeros(sol.y.shape[1])
        path = np.cumsum(np.sqrt(np.sum(np.diff(sol.y,axis=-1)**2, axis=0)))
        path = np.insert(path, 0, 0) 
        return magB, path

      

      B_max_particle = B

    

      if B_max_particle<B_max_point and B_max_particle>B_value and s0<1:
               rhs_B_forward = lambda l, y: eq.get_B(y)[1]/ np.linalg.norm(eq.get_B(y)[1])
               rhs_B_backward = lambda l, y: -eq.get_B(y)[1]/ np.linalg.norm(eq.get_B(y)[1])

               forward_magB, forward_path = solve_differential(point, L, N, rhs_B_forward)
               backward_magB, backward_path = solve_differential(point, L, N, rhs_B_backward)
               
               #-------------------Integral----------------------------
               def compute_integrals(magB, path, mu, B_max_particle):
                 mask = magB <= B_max_particle
                 idx_limit = np.argmax(~mask) if np.any(~mask) else len(magB)
                 magB_limited, path_limited = magB[:idx_limit], path[:idx_limit]
                 integrand = np.sqrt(2 * np.abs(mu) * (B_max_particle - magB_limited))
                 return integrand, path_limited
    

               forward_integrand, forward_path_limited = compute_integrals(forward_magB, forward_path, mu_value, B_max_particle)
               backward_integrand, backward_path_limited = compute_integrals(backward_magB, backward_path, mu_value, B_max_particle)
              

               complete_path = np.concatenate([
                backward_path_limited[::-1],  
                forward_path_limited,         
                forward_path_limited[::-1],   
                backward_path_limited         
                ])

               complete_integrand = np.concatenate([
                 backward_integrand[::-1],    
                 forward_integrand,           
                 forward_integrand[::-1],     
                 backward_integrand          
                 ])
               
               J_0 = trapezoidal_integral(complete_integrand, complete_path)*1e6
      else:
               J_0 = np.nan    

      return J_0


def MagField(points, t):
    previous_directory = os.getcwd()
    try:
        os.chdir('J_0_test')
        mconf_config = {
            'B0': 2.525,
            'B0_angle': 0.0,
            'accuracy': 1e-10, 
            'truncation': 1e-10
        }
        eq = mconf.Mconf_equilibrium(t, mconf_config=mconf_config)

        B_array, B_vec_array, S_array, B_max_array = [], [], [], []
        for point in tqdm(points, desc="Calculating Magnetic Field"):
            S, vecB = eq.get_B(point)
            B_max = eq.get_Bmax(S)
            valueB = np.sqrt(vecB[0]**2 + vecB[1]**2 + vecB[2]**2)

            if S<=1:
             B_array.append(valueB)
             B_vec_array.append(vecB)
             S_array.append(S)
             B_max_array.append(B_max)
            else:
             B_array.append(np.nan)
             B_vec_array.append(np.zeros_like(vecB))
             S_array.append(np.nan)
             B_max_array.append(np.nan)

    finally:
        os.chdir(previous_directory)

    return np.array(B_array), np.array(B_vec_array), np.array(S_array), np.array(B_max_array)


if __name__ == "__main__":

    # Загрузка данных
    Phi, R_phi, Z_phi = FuD.read_data()
    Phi, R_phi, Z_phi = Phi[17], R_phi[17], Z_phi[17]
    print(Phi)

    contour = np.array(transform(R_phi, Z_phi, Phi)).T
    path = Path(contour)


    R_min, R_max = min(R_phi) + 1, max(R_phi) - 1
    Z_min, Z_max = min(Z_phi) + 1, max(Z_phi) - 1
    grid_R, grid_Z = np.meshgrid(np.linspace(R_min, R_max, 30),
                                 np.linspace(Z_min, Z_max, 40))


    grid_points = np.vstack((grid_R.ravel(), grid_Z.ravel())).T
    mask = path.contains_points(grid_points).reshape(grid_R.shape)


    R_inside = grid_R[mask]
    Z_inside = grid_Z[mask]


    X_inside, Y_inside, Z_inside_3D = [], [], []
    for r, z in zip(R_inside, Z_inside):
        x, y, z = inverse_transform(r, z, Phi)
        X_inside.append(x)
        Y_inside.append(y)
        Z_inside_3D.append(z)
    points_inside = np.vstack((X_inside, Y_inside, Z_inside_3D)).T

    #b2 = 'w7x-sc1_ecrh_beta=0.02.bc'
    b0 = 'w7x-sc1.bc'
    b4 = 'w7x-sc1_ecrh_beta=0.04.bc'
    B_array_1, B_vec_array, S_array, B_max_array_1 = MagField(points_inside/100, b0)
    B_array_2, B_vec_array, S_array, B_max_array_2 = MagField(points_inside/100, b4)
    
    
    minim = max(np.nanmax(B_array_1), np.nanmax(B_array_2))
    maxim = min(np.nanmin(B_max_array_1), np.nanmin(B_max_array_2))
    print(np.nanmax(B_array_1))
    print(np.nanmax(B_array_2))
    print(np.nanmin(B_max_array_1))
    print(np.nanmin(B_max_array_2))


    B = 2.5#(minim + maxim) / 2
    print(B)

    #B_1 = (np.max(B_array_1) + np.min(B_max_array_1))/2
    #B_2 = (np.max(B_array_2) +np.min(B_max_array_2) )/2


    res_1 = J_0_calculate(points_inside/100,b0, B)
    res_1 = np.where(res_1==0, np.nan, res_1)

    res_2 = J_0_calculate(points_inside/100,b4, B)
    res_2 = np.where(res_2==0, np.nan, res_2)




    B_grid_1, B_grid_2  = np.full(grid_R.shape, np.nan), np.full(grid_R.shape, np.nan)
    J_0_grid_1,J_0_grid_2   = np.full(grid_R.shape, np.nan),np.full(grid_R.shape, np.nan)

    B_grid_1[mask]  = B_array_1
    J_0_grid_1[mask] = res_1

    B_grid_2[mask]  = B_array_2
    J_0_grid_2[mask] = res_2






    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)


    contour1 = axes[0].contour(grid_R, grid_Z, B_grid_1, levels=20, cmap="plasma")
    axes[0].plot(R_phi, Z_phi, color="red", linewidth=1)
    axes[0].set_title('w7x-sc1.bc')

    contour2 = axes[1].contour(grid_R, grid_Z, J_0_grid_1, levels=20, cmap="plasma")
    axes[1].plot(R_phi, Z_phi, color="red", linewidth=1)
    axes[1].set_title('J_0')

    contour3 = axes[2].contour(grid_R, grid_Z, B_grid_2, levels=20, cmap="plasma")
    axes[2].plot(R_phi, Z_phi, color="red", linewidth=1)
    axes[2].set_title('w7x-sc1_ecrh_beta=0.04.bc')

    contour4 = axes[3].contour(grid_R, grid_Z, J_0_grid_2, levels=20 , cmap="plasma")
    axes[3].plot(R_phi, Z_phi, color="red", linewidth=1)
    axes[3].set_title('J_0')






    fig.subplots_adjust(right=0.80)
    fig.colorbar(contour1, ax=axes[0])
    fig.colorbar(contour2, ax=axes[1])
    fig.colorbar(contour3, ax=axes[2])
    fig.colorbar(contour4, ax=axes[3])
    plt.show()
