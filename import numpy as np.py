import numpy as np
import matplotlib.pyplot as plt

class GraphPlotter:
    def __init__(self, num_arrays=9):  # Параметр num_arrays для определения размера матрицы
        self.num_arrays = num_arrays
        self.Name_NBI = [f"NBI{i}" for i in range(num_arrays)]  # Примеры имен NBI
        self.Name_Ports = [f"Port{i}" for i in range(num_arrays)]  # Примеры имен портов

    def draw_random_graphs(self):
        color = np.array([])
        Matr = np.empty((self.num_arrays, self.num_arrays), dtype=object)

        # Создаем фигуру matplotlib
        fig, axs = plt.subplots(self.num_arrays, self.num_arrays, figsize=(8, 8))

        for i in range(self.num_arrays):
            for j in range(self.num_arrays):
                MATRIX = np.random.rand(10, 10)  # Генерация случайной матрицы 10x10
                min_value = np.min(MATRIX)
                color = np.append(color, min_value)
                Matr[i, j] = MATRIX

        for i in range(self.num_arrays):
            for j in range(self.num_arrays):
                One_Matr = Matr[i, j]
                im = axs[i, j].imshow(One_Matr, cmap='Blues', origin='upper', aspect='auto', vmin=np.min(color), vmax=1.0)

                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        plt.subplots_adjust(wspace=0, hspace=0)

        # Добавление цветовой полосы к последнему подграфику
        cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [x, y, width, height]
        plt.colorbar(im, cax=cax)

        # Установка подписей осей
        for i in range(self.num_arrays):
            fonts = 6 if self.num_arrays >= 11 else 9
            selected_nbi = self.Name_NBI[i]
            selected_port = self.Name_Ports[i]
            name = 'S' if selected_nbi[0] == 'N' else 'C'
            axs[self.num_arrays - 1, i].set_xlabel(f'{selected_port}.{name}{selected_nbi}', fontsize=fonts)
            axs[i, 0].set_ylabel(f'{selected_port}.{name}{selected_nbi}', fontsize=fonts)

        plt.show()  # Отображаем график

# Пример использования класса
plotter = GraphPlotter(num_arrays=5)
plotter.draw_random_graphs()
