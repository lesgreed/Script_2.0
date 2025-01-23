import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid as cumtrapz
import matplotlib.pyplot as plt
import mconf.mconf as mconf
import os

# Физические константы
m = 1.65 * 10**(-27)  # масса частицы

# Задание магнитной конфигурации
os.chdir('J_0_test')

mconf_config = {
    'B0': 2.525,
    'B0_angle': 0.0,
    'accuracy': 1e-10,  # точность трансформации
    'truncation': 1e-10  # отсечение гармоник
}
eq = mconf.Mconf_equilibrium('w7x-sc1.bc', mconf_config=mconf_config)

# Исходная точка
point = [-0.10411962, 5.01378673, 0.19537387]

# Получение начального состояния
s0, vecB = eq.get_B(point)
r1 = np.array(eq.mag2xyz(s0, 0., 0.))

# Решение уравнения движения
L = 1  # длина интегрирования
N = 10000  # количество шагов
rhs_B = lambda l, y: eq.get_B(y)[1] / np.linalg.norm(eq.get_B(y)[1])

# Прямой и обратный путь
field_line_sol_forward = solve_ivp(rhs_B, [0, L], r1, method='RK45', max_step=L / N, atol=1e-6, dense_output=True)
field_line_sol_backward = solve_ivp(rhs_B, [0, -L], r1, method='RK45', max_step=L / N, atol=1e-6, dense_output=True)

# Вычисление величин вдоль траектории
def compute_B_along_path(solution):
    x, y, z = solution.y
    s, B = eq.get_s_B_T(x, y, z)
    magB = np.linalg.norm(B, axis=1)
    path = np.hstack([0, np.cumsum(np.sqrt(np.sum((np.diff(solution.y, axis=-1) ** 2), axis=0)))])
    return path, magB

path_forward, magB_forward = compute_B_along_path(field_line_sol_forward)
path_backward, magB_backward = compute_B_along_path(field_line_sol_backward)

# Параметры интегрирования
E_values = np.linspace(10, 100, 100) * (1.6 * 10**(-19)) * 10**3  # энергии (от 10 кэВ до 100 кэВ)
mu_values = np.linspace(0.1, 100, 100) * (1.6 * 10**(-19)) * 10**3  # магнитные моменты

J_0_map = np.zeros((len(E_values), len(mu_values)))

# Вычисление J_0 для каждой комбинации (E, mu)
for i, E in enumerate(E_values):
    for j, mu in enumerate(mu_values):
        # Определяем B_max для данного E и \mu
        B_max = E / mu
        
        # Поиск индексов B_max на траектории
        idx_forward = np.where(magB_forward >= B_max)[0]
        print(idx_forward)
        idx_backward = np.where(magB_backward >= B_max)[0]

        # Проверка наличия запертых точек
        if len(idx_forward) < 2 or len(idx_backward) < 2:
            J_0_map[i, j] = 0
            continue

        # Определяем границы интегрирования
        start_idx_forward = idx_forward[0]
        end_idx_forward = idx_forward[-1]
        start_idx_backward = idx_backward[0]
        end_idx_backward = idx_backward[-1]

        path_contour = np.concatenate([
            path_forward[start_idx_forward:end_idx_forward],
            path_backward[start_idx_backward:end_idx_backward][::-1]
        ])
        magB_contour = np.concatenate([
            magB_forward[start_idx_forward:end_idx_forward],
            magB_backward[start_idx_backward:end_idx_backward][::-1]
        ])

        # Интеграция J_0
        integrand = np.sqrt(2 * mu * (B_max - magB_contour))
        J_0_map[i, j] = cumtrapz(integrand, x=path_contour, initial=0)[-1]

# Визуализация
plt.figure(figsize=(8, 6))
plt.contourf(E_values / (1.6 * 10**(-19)) / 10**3, 
             mu_values / (1.6 * 10**(-19)) / 10**3, 
             J_0_map.T / m, levels=50, cmap='viridis')
plt.colorbar(label=r'$J_0$ (м·Тл)')
plt.xlabel(r'$E$ (кэВ)')
plt.ylabel(r'$\mu$ ($10^{-3} \cdot \text{eV/T}$)')
plt.title('Зависимость $J_0$ от $E$ и $\mu$')
plt.show()
