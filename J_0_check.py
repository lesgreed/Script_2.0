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


def trapezoidal_integral(f, s):
    """
    Интегрирование функции f(s) по пути, используя метод трапеций.

    Параметры:
    ----------
    s : ndarray
        Массив расстояний от нулевой точки.
    f : ndarray
        Массив значений функции f(s) в точках s.

    Возвращает:
    ----------
    integral : float
        Результат интегрирования.
    """
    # Проверяем, что размеры массивов совпадают
    if len(s) != len(f):
        raise ValueError("Длины массивов s и f должны совпадать")

    # Вычисляем разности между соседними точками s
    ds = np.abs(np.diff(s))

    # Средние значения функции между соседними точками
    avg_f = (f[:-1] + f[1:]) / 2

    # Интеграл на каждом сегменте
    segment_integrals = ds * avg_f

    # Суммируем все сегментные интегралы
    integral = np.sum(segment_integrals)
    
    return integral



def J_0_calculate(points):

    points = np.array(points, dtype=np.float64)/ 100  
    previous_directory = os.getcwd()

    try:

        os.chdir('J_0_test')
        mconf_config = {
            'B0': 2.525,
            'B0_angle': 0.0,
            'accuracy': 1e-10,  
            'truncation': 1e-10  }

        eq = mconf.Mconf_equilibrium('w7x-sc1_ecrh_beta=0.04.bc', mconf_config=mconf_config)
        J_0_all = np.zeros(len(points))  
        for i, point in tqdm(enumerate(points), total=len(points)):
         s0, vecB = eq.get_B(point)
         B_value = np.linalg.norm(vecB)
         B_max_p = eq.get_Bmax(s0)
         B_max = E/mu
         print("s0: ",s0)
         if B_max<B_max_p and B_max>B_value and s0<1+1e-4:
          #if B_max>

            try:                   
                 L = 100  
                 N = 2000  
                 rhs_B = lambda l, y: eq.get_B(y)[1]/ np.linalg.norm(eq.get_B(y)[1])
                 rhs_B_backward = lambda l, y: -eq.get_B(y)[1]/ np.linalg.norm(eq.get_B(y)[1])

                 forward_sol = solve_ivp(rhs_B, [0, L], point,method='RK45', max_step=L / N,atol=1e-6, dense_output=True)
                 forward_s, forward_B = eq.get_s_B_T(forward_sol.y[0],forward_sol.y[1], forward_sol.y[2])
                 forward_magB = np.linalg.norm(forward_B, axis=1)
                 forward_path = np.zeros(forward_sol.y.shape[1])
                 forward_path = np.cumsum(np.sqrt(np.sum(np.diff(forward_sol.y,axis=-1)**2, axis=0)))
                 forward_path = np.insert(forward_path, 0, 0)                   

                 backward_sol = solve_ivp(rhs_B_backward, [0, L], point,method='RK45', max_step=L / N, atol=1e-6, dense_output=True)
                 backward_s, backward_B = eq.get_s_B_T(backward_sol.y[0],backward_sol.y[1], backward_sol.y[2])
                 backward_magB = np.linalg.norm(backward_B, axis=1)
                 backward_path = np.zeros(backward_sol.y.shape[1])
                 backward_path = np.cumsum(np.sqrt(np.sum(np.diff(backward_sol.y,axis=-1)**2, axis=0)))
                 backward_path = np.insert(backward_path, 0, 0)  

                 forward_mask = forward_magB <= B_max
                 forward_idx_limit = np.argmax(~forward_mask) if np.any(~forward_mask) else len(forward_magB)
                 forward_magB_limited = forward_magB[:forward_idx_limit]
                 forward_path_limited = forward_path[:forward_idx_limit]
                 print("forward_idx_limit", forward_idx_limit)
                 forward_integrand = np.sqrt(2 * mu* (B_max - forward_magB_limited))

                 backward_mask = backward_magB <= B_max
                 backward_idx_limit = np.argmax(~backward_mask) if np.any(~backward_mask) else len(backward_magB)
                 backward_magB_limited = backward_magB[:backward_idx_limit]
                 backward_path_limited = backward_path[:backward_idx_limit]
                 print("backward_idx_limit", backward_idx_limit)
                 backward_integrand = np.sqrt(2 * mu* (B_max - backward_magB_limited))

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
                 J_0 = trapezoidal_integral(complete_integrand, complete_path)
                 J_0_all[i] = J_0
            except Exception as e:
                J_0_all[i] = 0
         else:
            J_0_all[i] = 0
         print(f"Point{i}: J_0 = {J_0_all[i]}")
    finally:
        os.chdir(previous_directory)
    return J_0_all




def MagField(points):
      previous_directory = os.getcwd()
      os.chdir('J_0_test')
      mconf_config = {'B0': 2.525,
                'B0_angle': 0.0,
                'accuracy': 1e-8, #accuracy of magnetic to cartesian coordinat transformation
                'truncation': 1e-8} #trancation of mn harmonics
      eq = mconf.Mconf_equilibrium('w7x-sc1_ecrh_beta=0.04.bc',mconf_config=mconf_config)
      B_array, B_vec_array, S_array, B_max_array= [], [], [], []
      for i in range(len(points)):
         S, vecB = eq.get_B(points[i])
         B_max = eq.get_Bmax(S)
         valueB = np.sqrt(vecB[0]**2 + vecB[1]**2 + vecB[2]**2)
         B_array.append(valueB)
         B_vec_array.append(vecB)
         S_array.append(S)
         B_max_array.append(B_max)

      os.chdir(previous_directory)
      return B_array, B_vec_array, S_array, B_max_array

if __name__ == "__main__":
    # Константы для расчета
    E = 75
    mu = 30
    print(f"E/mu = {E / mu}")

    # Загрузка данных
    Phi, R_phi, Z_phi = FuD.read_data()
    Phi, R_phi, Z_phi = Phi[25], R_phi[25], Z_phi[25]

    # Преобразование точек контура в двумерный массив для построения
    contour = np.array(transform(R_phi, Z_phi, Phi)).T
    path = Path(contour)

    # Создание регулярной сетки в координатах R, Z
    R_min, R_max = min(R_phi) - 1, max(R_phi) + 1
    Z_min, Z_max = min(Z_phi) - 1, max(Z_phi) + 1
    grid_R, grid_Z = np.meshgrid(np.linspace(R_min, R_max, 50),

                                 np.linspace(Z_min, Z_max, 50))

    # Проверка точек сетки на принадлежность контуру
    grid_points = np.vstack((grid_R.ravel(), grid_Z.ravel())).T
    mask = path.contains_points(grid_points).reshape(grid_R.shape)

    # Выделение точек внутри контура
    R_inside = grid_R[mask]
    Z_inside = grid_Z[mask]

    # Преобразование точек в декартовы координаты для расчета J_0
    X_inside, Y_inside, Z_inside_3D = [], [], []
    for r, z in zip(R_inside, Z_inside):
        x, y, z = inverse_transform(r, z, Phi)
        X_inside.append(x)
        Y_inside.append(y)
        Z_inside_3D.append(z)
    points_inside = np.vstack((X_inside, Y_inside, Z_inside_3D)).T

    # Расчет значений J_0 внутри контура
    print(len(points_inside))
    J_0_values = J_0_calculate(points_inside)
    J_0_grid = np.full(grid_R.shape, np.nan)
    J_0_grid[mask] = J_0_values

    # Построение графика
    plt.figure(figsize=(10, 8))

    # Линии уровня для J_0, пропуская NaN
    contour = plt.contour(grid_R, grid_Z, J_0_grid, levels=50, cmap="Blues")

    plt.clabel(contour, inline=True, fontsize=6)  # Добавление подписей к линиям уровня
    # Добавление контура
    plt.plot(R_phi, Z_phi, color="blue", label="Contour", linewidth=2)

# Подписи и легенда
    plt.xlabel("R")
    plt.ylabel("Z")
    plt.title("Function J_0 in R-Z Coordinates")
    plt.legend()
    plt.grid(True)

# Отображение графика
    plt.show()

















