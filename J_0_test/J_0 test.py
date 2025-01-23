import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid as cumtrapz

import matplotlib.pyplot as plt
import mconf.mconf as mconf
import os
m = 1.65 * 10**(-27) 

# Задание магнитной конфигурации
os.chdir('J_0_test')

mconf_config = {
    'B0': 2.525,
    'B0_angle': 0.0,
    'accuracy': 1e-10,  # точность трансформации
    'truncation': 1e-10  # отсечение гармоник
}
eq = mconf.Mconf_equilibrium('wout_EIM_0.txt', mconf_config=mconf_config)

# Исходная точка
point = [-0.10411962, 5.01378673, 0.19537387]

# Получение начального состояния
s0, vecB = eq.get_B(point)
r1 = np.array(eq.mag2xyz(s0, 0., 0.))

# Решение уравнения движения
L = 20  # длина интегрирования
N = 10000  # количество шагов
rhs_B = lambda l, y: eq.get_B(y)[1] / np.linalg.norm(eq.get_B(y)[1])
field_line_sol = solve_ivp(rhs_B, [0, L], r1, method='RK45', max_step=L / N, atol=1., dense_output=True)

# Вычисление величин вдоль траектории
s, B = eq.get_s_B_T(field_line_sol.y[0], field_line_sol.y[1], field_line_sol.y[2])
magB = np.linalg.norm(B, axis=1)
path = np.hstack([0, np.cumsum(np.sqrt(np.sum((np.diff(field_line_sol.y, axis=-1) ** 2), axis=0)))])

# Параметры интегрирования
E_values = np.linspace(10, 100, 100)*(1.6*10**(-19))*10**3  # массив энергий (например, от 1 мэВ до 10 эВ)
mu_values = np.linspace(0, 100, 100)*(1.6*10**(-19))*10**3  # массив магнитных моментов (в единицах \mu)

J_0_map = np.zeros((len(E_values), len(mu_values)))

# Вычисление J_0 для каждой комбинации (E, mu)
for i, E in enumerate(E_values):
    for j, mu in enumerate(mu_values):
        # Определяем допустимые точки (где частица заперта)
        B_max = E / mu  # Максимальное магнитное поле для данного E и \mu
        mask = magB <= B_max

        # Если траектория не имеет запертых точек, пропускаем
        if not np.any(mask):
            J_0_map[i, j] = 0
            continue

        # Интегрирование J_0
        integrand = np.sqrt(2 * mu * (B_max - magB[mask]))
        print(magB)
        J_0_map[i, j] = cumtrapz(integrand, x=path[mask], initial=0)[-1]

# Параметры интегрирования
E_values = np.linspace(10, 100, 100)  # массив энергий (например, от 1 мэВ до 10 эВ)
mu_values = np.linspace(0.1, 100, 100)  # массив магнитных моментов (в единицах \mu)

# Визуализация
plt.figure(figsize=(8, 6))
plt.contourf(E_values, mu_values, (J_0_map.T)/m, levels=50, cmap='viridis')
plt.colorbar(label=r'$J_0$ (m·T)')
plt.xlabel(r'$E$ (keV)')
plt.ylabel(r'$mu$ ')
plt.title(' $J_0$($E$,$\mu$)')
plt.show()
