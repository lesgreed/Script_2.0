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

def transform(R, Z, phi):
    x = R   # Use R directly for x-axis
    y = Z  # Use Z directly for y-axis
    return x, y

def inverse_transform(R, Z, Phi):
    # Convert R, Z, and Phi back to x, y, z
    phi_radian = np.radians(Phi)
    x = R * np.cos(phi_radian)
    y = R * np.sin(phi_radian)
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
    points = np.array(points, dtype=np.float64) / 100  
    previous_directory = os.getcwd()


    
    os.chdir('J_0_test')

    mconf_config = {
            'B0': 2.525,
            'B0_angle': 0.0,
            'accuracy': 1e-2,  
            'truncation': 1e-2  
        }

    eq = mconf.Mconf_equilibrium('wout_EIM_0.txt', mconf_config=mconf_config)
    J_0_all = np.zeros(len(points))  
    L = 20; N = 10000; 
    from scipy.integrate import solve_ivp
    rhs_B_fortran1 = lambda t, y:  eq.get_B(y)[1]

    r1 = np.array(eq.mag2xyz(0.1,0.,0.))
    print(r1)
    print(eq.mag_B(*r1))
       

    return J_0_all



def MagField(points):
      previous_directory = os.getcwd()
      os.chdir('J_0_test')
      mconf_config = {'B0': 2.525,
                'B0_angle': 0.0,
                'accuracy': 1e-8, #accuracy of magnetic to cartesian coordinat transformation
                'truncation': 1e-8} #trancation of mn harmonics
      eq = mconf.Mconf_equilibrium('w7x-sc1_ecrh_beta=0.02.bc',mconf_config=mconf_config)
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
    E = 80 
    mu = 32
    print(f"E/mu = {E / mu}")

    # Загрузка данных
    Phi, R_phi, Z_phi = FuD.read_data()
    Phi, R_phi, Z_phi = Phi[20], R_phi[20], Z_phi[20]

    # Преобразование точек контура в двумерный массив для построения
    contour = np.array(transform(R_phi, Z_phi, Phi)).T
    path = Path(contour)

    # Создание регулярной сетки в координатах R, Z
    R_min, R_max = min(R_phi) - 1, max(R_phi) + 1
    Z_min, Z_max = min(Z_phi) - 1, max(Z_phi) + 1
    grid_R, grid_Z = np.meshgrid(np.linspace(R_min, R_max, 10),
                                 np.linspace(Z_min, Z_max, 10))

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
        x, y, z = x, y, z
        X_inside.append(x)
        Y_inside.append(y)
        Z_inside_3D.append(z)
    points_inside = np.vstack((X_inside, Y_inside, Z_inside_3D)).T

    # Расчет значений J_0 внутри контура
    print(len(points_inside))
    J_0_values = J_0_calculate(points_inside)

    # Преобразование J_0 в формат для сетки
    J_0_grid = np.full(grid_R.shape, np.nan)
    J_0_grid[mask] = J_0_values

    # Построение графика
    plt.figure(figsize=(10, 8))

    # Линии уровня для J_0
    contour = plt.contour(grid_R, grid_Z, J_0_grid, levels=200, cmap="Blues")#, linewidths = 1)  # Контуры J_0
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

