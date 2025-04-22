import numpy as np

# Сетка по x и y
x = np.linspace(2, 30, 400)
y = np.linspace(-100, 100, 400)



# Генерация случайной матрицы
random_matrix = np.random.rand(400, 400) * 10

# Шаги по x и y
x, y = np.meshgrid(x, y)
dx = x[0, 1] - x[0, 0]  
dy = y[1, 0] - y[0, 0]  


x = np.linspace(2, 30, 400)
y = np.linspace(-100, 100, 400)


# Интегрирование по двумерной сетке методом прямоугольников
integral = np.sum(random_matrix) * dx * dy

# Интегрирование методом трапеций по оси x (по строкам)
integral_trapz_x = np.trapz(random_matrix, x, axis=1)  # Интегрируем по оси x (для каждой строки)

# Интегрирование результата по оси y (по столбцам)
integral_trapz = np.trapz(integral_trapz_x, y)  # Интегрируем результат по оси y

print(f"Интеграл методом прямоугольников: {integral}")
print(f"Интеграл методом трапеций: {integral_trapz}")
