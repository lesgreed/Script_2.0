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




def MagField(points, B0, config):
    previous_directory = os.getcwd()
    try:
        os.chdir('J_0_test')
        mconf_config = {
            'B0': B0,
            'B0_angle': 0.0,
            'accuracy': 1e-10, 
            'truncation': 1e-10
        }
        eq = mconf.Mconf_equilibrium(config, mconf_config=mconf_config)

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

if __name__ == "__main__":

    # Загрузка данных
    Phi, R_phi, Z_phi = FuD.read_data()
    Phi, R_phi, Z_phi = Phi[0], R_phi[0], Z_phi[0]
    print(Phi)

    contour = np.array(transform(R_phi, Z_phi, Phi)).T
    path = Path(contour)


    R_min, R_max = min(R_phi) - 1, max(R_phi) + 1
    Z_min, Z_max = min(Z_phi) - 1, max(Z_phi) + 1
    grid_R, grid_Z = np.meshgrid(np.linspace(R_min, R_max, 60),
                                 np.linspace(Z_min, Z_max, 60))


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


    config = 'FTM_beta=0.txt'
    B0 = 2.520
    B_array_1, B_vec_array, S_array, B_max_array_1 = MagField(points_inside/100, B0, config)

    
    
    B_grid_1 = np.full(grid_R.shape, np.nan)
    B_grid_1[mask] = B_array_1

    fig, ax = plt.subplots(figsize=(6, 5))


    levels = np.linspace(2.0, 3.5, 40)
    contour = ax.contour(grid_R, grid_Z, B_grid_1, levels=levels, cmap="jet")

    ax.plot(R_phi, Z_phi, color="red", linewidth=2)
    ax.set_title(f'{config}, {B0}')
    ax.set_aspect('equal')
    fig.colorbar(contour, ax=ax)
    
    plt.show()
    
    








