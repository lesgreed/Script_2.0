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
            B_array.append(valueB)
            B_vec_array.append(vecB)
            S_array.append(S)
            B_max_array.append(B_max)

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
    grid_R, grid_Z = np.meshgrid(np.linspace(R_min, R_max, 80),
                                 np.linspace(Z_min, Z_max, 80))


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

    b2 = 'w7x-sc1_ecrh_beta=0.02.bc'
    b0 = 'w7x-sc1.bc'
    b4 = 'w7x-sc1_ecrh_beta=0.04.bc'
    B_array_1, B_vec_array, S_array, B_max_array = MagField(points_inside/100, b0)
    B_array_2, B_vec_array, S_array, B_max_array = MagField(points_inside/100, b2)
    B_array_3, B_vec_array, S_array, B_max_array = MagField(points_inside/100, b4)



    B_grid_1,B_grid_2, B_grid_3  = np.full(grid_R.shape, np.nan), np.full(grid_R.shape, np.nan), np.full(grid_R.shape, np.nan)
    B_grid_1[mask],B_grid_2[mask], B_grid_3[mask]  = B_array_1, B_array_2, B_array_3



    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)

    # Определяем общий диапазон значений магнитного поля
    vmin = min(np.nanmin(B_grid_1), np.nanmin(B_grid_2), np.nanmin(B_grid_3))
    vmax = max(np.nanmax(B_grid_1), np.nanmax(B_grid_2), np.nanmax(B_grid_3))

    # Создаем контурные графики с одинаковыми уровнями
    contour1 = axes[0].contourf(grid_R, grid_Z, B_grid_1, levels=200, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[0].plot(R_phi, Z_phi, color="red", linewidth=1)
    axes[0].set_title('w7x-sc1.bc')

    contour2 = axes[1].contourf(grid_R, grid_Z, B_grid_2, levels=200, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[1].plot(R_phi, Z_phi, color="red", linewidth=1)
    axes[1].set_title('w7x-sc1_ecrh_beta=0.02.bc')

    contour3 = axes[2].contourf(grid_R, grid_Z, B_grid_3, levels=200, cmap="Blues", vmin=vmin, vmax=vmax)
    axes[2].plot(R_phi, Z_phi, color="red", linewidth=1)
    axes[2].set_title('w7x-sc1_ecrh_beta=0.04.bc')

    contour4 = axes[3].contourf(grid_R, grid_Z, B_grid_1 - B_grid_2, levels=200, cmap="plasma")
    axes[3].plot(R_phi, Z_phi, color="red", linewidth=1)
    axes[3].set_title('Difference (β=0.00 - β=0.02)')

        # Добавляем общий colorbar
    cbar = fig.colorbar(contour1, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label("Magnetic Field Strength (T)")

    fig.subplots_adjust(right=0.80)
    fig.colorbar(contour4, ax=axes[3])
    plt.show()
