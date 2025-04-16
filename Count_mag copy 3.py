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
            print(eq.get_B2avrg(S))
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


    R_min, R_max = min(R_phi) - 1, max(R_phi) + 1
    Z_min, Z_max = min(Z_phi) - 1, max(Z_phi) + 1
    grid_R, grid_Z = np.meshgrid(np.linspace(R_min, R_max, 5),
                                 np.linspace(Z_min, Z_max, 5))


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
    center_R = 518.9716486718739
    center_Z = 4.459381879916724
    Phi = 34.0
    point = np.array(inverse_transform(center_R, center_Z, Phi))
    print(point)
    point = point/100

    os.chdir('J_0_test')
    mconf_config = {
            'B0': 2.520,
            'B0_angle': 0.0,
            'accuracy': 1e-10, 
            'truncation': 1e-10
        }
    eq = mconf.Mconf_equilibrium("w7x-sc1_ecrh_beta=0.02.bc", mconf_config=mconf_config)
    
    s0, vecB = eq.get_B(point)
    B_ref = 2.48969838045484
    
    fig = plt.figure(figsize=(5,6))
    ax1 = fig.add_subplot(111)
    phi = np.linspace(0,2*np.pi,20)
    T = np.linspace(0,2*np.pi,40)
    Ts,Ps = np.meshgrid(phi,T)
    X,Y,Z = eq.mag2xyz(0.8,Ts,Ps)
    s, B_lab  = eq.get_s_B(X,Y,Z)
    modB      = np.linalg.norm(B_lab,axis=2)
    ax1.contour(T, phi, modB.T, levels=[B_ref])

    ax1.grid()
    ax1.axis('equal')
    plt.show()
    



    

