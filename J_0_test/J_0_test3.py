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
    'max_iterations': 2000,
    'accuracy': 1e-8,  # точность трансформации
    'truncation': 1e-8  # отсечение гармоник
    
}
eq = mconf.Mconf_equilibrium('w7x-sc1.bc', mconf_config=mconf_config)

# Исходная точка
point = [0.28664648, 6.23121216, 0.30403593]

# Получение начального состояния
s0, vecB = eq.get_B(point)
r1 = np.array(eq.mag2xyz(s0, 0., 0.))

# Решение уравнения движения вдоль силовой линии
L = 20  # длина интегрирования
N = 1000  # количество шагов
rhs_B = lambda l, y: eq.get_B(y)[1] / np.linalg.norm(eq.get_B(y)[1])
field_line_sol = solve_ivp(rhs_B, [0, L], r1, method='RK45', max_step=L / N, atol=1e-6, dense_output=True)

# Вычисление величин вдоль траектории
s, B = eq.get_s_B_T(field_line_sol.y[0], field_line_sol.y[1], field_line_sol.y[2])
magB = np.linalg.norm(B, axis=1)
path = np.hstack([0, np.cumsum(np.sqrt(np.sum((np.diff(field_line_sol.y, axis=-1) ** 2), axis=0)))])

# Параметры интегрирования
E_values = np.linspace(10, 100, 100) * (1.6 * 10**(-19)) * 10**3  # энергии (от 10 кэВ до 100 кэВ)
mu_values = np.linspace(0.1, 100, 100) * (1.6 * 10**(-19)) * 10**3  # магнитные моменты

J_0_map = np.zeros((len(E_values), len(mu_values)))

# Вычисление J_0 для каждой комбинации (E, mu)
for i, E in enumerate(E_values):
    for j, mu in enumerate(mu_values):
        # Вычисляем B_max для данной комбинации (E, mu)
        B_max = E / mu

        # Определяем точку, где B >= B_max
        mask = magB <= B_max
        if not np.any(mask):
            J_0_map[i, j] = 0
            continue
        

        # Индекс последней точки, где B <= B_max
        idx_limit = np.argmax(~mask) if np.any(~mask) else len(magB)
        
        B_initial = magB[0]  # Магнитное поле в начальной точке
        if E <= B_initial * mu:
            J_0_map[i, j] = 0
            continue

        # Ограничиваем путь и величины
        magB_limited = magB[:idx_limit]
        path_limited = path[:idx_limit]

        # Интегрирование J_0
        integrand = np.sqrt(2 * mu * (B_max - magB_limited))
        J_0_map[i, j] = cumtrapz(integrand, x=path_limited, initial=0)[-1]

# Параметры для графика
E_values_keV = E_values / (1.6 * 10**(-19)) / 10**3  # перевод в кэВ
mu_values_normalized = mu_values / (1.6 * 10**(-19)) / 10**3  # перевод в единицы эВ/Т

# Визуализация
plt.figure(figsize=(8, 6))
plt.contourf(E_values_keV, mu_values_normalized, J_0_map.T , levels=50, cmap='viridis')
plt.colorbar(label=r'$J_0$ (м·Тл)')
plt.xlabel(r'$E$ (кэВ)')
plt.ylabel(r'$\mu$ ($10^{-3} \cdot \text{eV/T}$)')
plt.title('Зависимость $J_0$ от $E$ и $\mu$')
plt.show()
