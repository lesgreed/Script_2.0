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

def plot_zoomed_circle_view(grid_R, grid_Z, B_grid_1, B_grid_2, R_phi, Z_phi, R_inside, Z_inside, num_points=200, circle_radius=2):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # Центр круга
    center_R = np.mean(R_inside)-25
    center_Z = np.mean(Z_inside)
    print(center_R, Phi, center_Z)
    print("hi", inverse_transform(center_R, center_Z, Phi))
    center = np.array([center_R, center_Z])
    radius = circle_radius

    # Создание равномерной сетки точек
    r_vals = np.linspace(center_R - radius, center_R + radius, int(np.sqrt(num_points)*2))
    z_vals = np.linspace(center_Z - radius, center_Z + radius, int(np.sqrt(num_points)*2))
    R_grid_local, Z_grid_local = np.meshgrid(r_vals, z_vals)
    R_flat = R_grid_local.flatten()
    Z_flat = Z_grid_local.flatten()

    # Маска для точек внутри круга
    distances = np.sqrt((R_flat - center_R)**2 + (Z_flat - center_Z)**2)
    inside_mask = distances <= radius
    R_circle = R_flat[inside_mask]
    Z_circle = Z_flat[inside_mask]

    # === ПЛОТ ===
    fig, axes = plt.subplots(1, 1, figsize=(30, 5), sharex=False, sharey=False)

    # Контуры
    contour1 = axes[0].contourf(grid_R, grid_Z, B_grid_1, levels=20, cmap="plasma")
    axes[0].plot(R_phi, Z_phi, color="red", linewidth=0.5)
    axes[0].set_title('beta=1.txt')
    fig.colorbar(contour1, ax=axes[0])

    #contour3 = axes[1].contourf(grid_R, grid_Z, B_grid_2, levels=20, cmap="plasma")
    #axes[1].plot(R_phi, Z_phi, color="red", linewidth=0.5)
    #axes[1].set_title('beta=0.65.txt')

    # Круг на графиках 0 и 2
    #for ax in [axes[0], axes[1]]:
    #    circle = Circle((center_R, center_Z), radius, color='blue', fill=False, linestyle='--', linewidth=1.5)
    #    ax.add_patch(circle)
    plt.show()


    # Цветовые шкалы
    fig.subplots_adjust(right=0.80)

    return R_circle, Z_circle, R_grid_local, Z_grid_local, inside_mask
    


def calculate_relative_differences_1d(J_vals, coords_3d):
    min_diffs, max_diffs, distances = [], [], []

    for i, center in enumerate(J_vals):
        if np.isnan(center) or center == 0:
            continue

        center_coords = coords_3d[i]
        local_diffs = []
        local_dists = []

        for j, neighbor_val in enumerate(J_vals):
            if i == j or np.isnan(neighbor_val) or neighbor_val == 0:
                continue
            diff =100* np.abs(center - neighbor_val) / np.abs(center)
            dist = np.linalg.norm(center_coords - coords_3d[j])
            local_diffs.append(diff)
            local_dists.append(dist)
            


        if local_diffs:
            min_diffs.append(np.min(local_diffs))
            max_diffs.append(np.max(local_diffs))
            distances.extend(local_dists)

    return np.array(min_diffs), np.array(max_diffs), np.array(distances)



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
      mconf_config = {'B0': 2.911,
                'B0_angle': 0.0,
                'accuracy': 1e-10, #accuracy of magnetic to cartesian coordinat transformation
                'truncation': 1e-10} #trancation of mn harmonics
      eq = mconf.Mconf_equilibrium(config,mconf_config=mconf_config)
      

      #constant
      s0, vecB = eq.get_B(point)
      B_value = np.linalg.norm(vecB)
      B_max_point = eq.get_Bmax(s0)
      L = 300 
      N = 1000
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
            'B0': 2.911,
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
def mean_without_outliers_iqr(arr):
    arr = np.array(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered = arr[(arr >= lower_bound) & (arr <= upper_bound)]
    return np.mean(filtered)

if __name__ == "__main__":

    # Загрузка данных
    Phi, R_phi, Z_phi = FuD.read_data()
    Phi, R_phi, Z_phi = Phi[17], R_phi[17], Z_phi[17]
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


    b0 = 'beta=1.txt'
    b4 = 'beta=0.65.txt'
    B_array_1, B_vec_array, S_array, B_max_array_1 = MagField(points_inside/100, b0)
    B_array_2, B_vec_array, S_array, B_max_array_2 = MagField(points_inside/100, b4)
    
    
    minim = max(np.nanmax(B_array_1), np.nanmax(B_array_2))
    maxim = min(np.nanmin(B_max_array_1), np.nanmin(B_max_array_2))
    print(np.nanmax(B_array_1))
    print(np.nanmax(B_array_2))
    print(np.nanmin(B_max_array_1))
    print(np.nanmin(B_max_array_2))


    B = 2.90
    print(B)


    #res_1 = J_0_calculate(points_inside/100,b4, B)
    #res_1 = np.where(res_1==0, np.nan, res_1)

    #res_2 = J_0_calculate(points_inside/100,b4, B)
    #res_2 = np.where(res_2==0, np.nan, res_2)




    B_grid_1, B_grid_2  = np.full(grid_R.shape, np.nan), np.full(grid_R.shape, np.nan)
    J_0_grid_1,J_0_grid_2   = np.full(grid_R.shape, np.nan),np.full(grid_R.shape, np.nan)

    B_grid_1[mask]  = B_array_1
    #J_0_grid_1[mask] = res_1

    B_grid_2[mask]  = B_array_2
    #J_0_grid_2[mask] = res_2



    R_circle, Z_circle, R_grid_local, Z_grid_local, inside_mask = plot_zoomed_circle_view(grid_R, grid_Z, B_grid_1, B_grid_2, R_phi, Z_phi, R_inside, Z_inside, num_points=500, circle_radius=7 )


    X_circ, Y_circ, Z_circ = [], [], []
    for r, z in zip(R_circle, Z_circle):
        x, y, z = inverse_transform(r, z, Phi)
        X_circ.append(x)
        Y_circ.append(y)
        Z_circ.append(z)
    points_circ = np.vstack((X_circ, Y_circ, Z_circ)).T

    # === 2. Считаем B и J_0 внутри круга ===
    res_1 = J_0_calculate(points_circ / 100, b0, B)
    res_2 = J_0_calculate(points_circ / 100, b4, B)


    # === 3. Переводим в сетки и считаем отличия ===
    res_1_grid = np.full(R_grid_local.shape, np.nan)
    res_2_grid = np.full(R_grid_local.shape, np.nan)
    coords_3d = np.stack((X_circ, Y_circ, Z_circ), axis=-1)

    res_1_grid[inside_mask.reshape(R_grid_local.shape)] = res_1
    res_2_grid[inside_mask.reshape(R_grid_local.shape)] = res_2

        # Используем обновлённую функцию
    print(res_1)
    print(res_2)
    min_diff_1, max_diff_1, dists_1 = calculate_relative_differences_1d(np.array(res_1), coords_3d)
    min_diff_2, max_diff_2, dists_2 = calculate_relative_differences_1d(np.array(res_2), coords_3d)


    # === 5. Статистика расстояний ===
    print("Минимальное расстояние между точками:", np.nanmin(dists_1))
    print("Максимальное расстояние между точками:", np.nanmax(dists_1))


    # === 4. Гистограмма различий ===
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12, 5))  # 1 строка, 2 столбца

    # Первая гистограмма
    axs[0,0].hist(res_1, bins=30, alpha=0.6, color='blue')
    axs[0,0].set_title('Config 1: beta=1.txt')
    axs[0,0].set_xlabel('J_0')  # можешь заменить на нужное название
    axs[0,0].set_ylabel('Y')  # и это тоже
    axs[0,0].grid(True)

    # Вторая гистограмма
    axs[0,1].hist(res_2, bins=40, alpha=0.6, color='orange')  # предполагаем, что у тебя есть res_2
    axs[0,1].set_title('Config 2: beta=0.65.txt')
    axs[0,1].set_xlabel('J_0')
    axs[0,1].set_ylabel('Y')
    axs[0,1].grid(True)
    
        # Первая гистограмма
    axs[1,0].hist(min_diff_1, bins=30, alpha=0.6, color='blue')
    axs[1,0].set_title('Config 1: beta=1.txt')
    axs[1,0].set_xlabel('J_0 %')  # можешь заменить на нужное название
    axs[1,0].set_ylabel('Y')  # и это тоже
    axs[1,0].grid(True)

    # Вторая гистограмма
    axs[1,1].hist(min_diff_2, bins=40, alpha=0.6, color='orange')  # предполагаем, что у тебя есть res_2
    axs[1,1].set_title('Config 2: beta=0.65.txt')
    axs[1,1].set_xlabel('J_0 %')
    axs[1,0].set_ylabel('Y')
    axs[1,1].grid(True)
    
    

    plt.tight_layout()
    plt.show()



