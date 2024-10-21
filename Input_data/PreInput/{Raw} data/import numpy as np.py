import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# Уравнение: w^2^3 - 5 * w^2^2 + 6 * w^2 + cos(ak) - 1 = 0
def equation(x, ak):
    return x**3 - 5 * x**2 + 6 * x + np.cos(ak) - 1

# Функция для нахождения всех действительных корней x для заданного a*k
def solve_for_x(ak):
    roots = []
    # Пробуем разные начальные приближения, чтобы найти все корни
    for guess in [2, 0, 3]:  # Попробуем несколько начальных приближений
        sol = root(equation, x0=guess, args=(ak,))
        if sol.success and not np.isnan(sol.x[0]):
            root_val = sol.x[0]
            if all(abs(root_val - r) > 1e-5 for r in roots):  # Убираем дубликаты
                roots.append(root_val)
    return roots

# Диапазон значений для a * k
ak_vals = np.linspace(0, np.pi, 500)

# Создадим массивы для трех решений w (если они существуют)
w1_vals, w2_vals, w3_vals = [], [], []

# Для каждого значения a * k находим все корни и берем sqrt для каждого корня
for ak in ak_vals:
    x_roots = solve_for_x(ak)
    w_roots = [np.sqrt(x) if x >= 0 else np.nan for x in x_roots]  # Берем только неотрицательные корни
    # Заполняем массивы w1, w2, w3 (или заполняем NaN, если решения нет)
    w1_vals.append(w_roots[0] if len(w_roots) > 0 else np.nan)
    w2_vals.append(w_roots[1] if len(w_roots) > 1 else np.nan)
    w3_vals.append(w_roots[2] if len(w_roots) > 2 else np.nan)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(ak_vals, w1_vals, label=r'$w_1$', color='b')
plt.plot(ak_vals, w2_vals, label=r'$w_2$', color='g')
plt.plot(ak_vals, w3_vals, label=r'$w_3$', color='r')
plt.xlabel(r'$a \cdot k$', fontsize=14)
plt.ylabel(r'$w$', fontsize=14)
plt.title(r'График $w$ в зависимости от $a \cdot k$', fontsize=16)
plt.grid(True)
plt.legend()
plt.show()
