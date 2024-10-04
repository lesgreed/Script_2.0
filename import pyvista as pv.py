import pyvista as pv
import numpy as np

# Пример функции для обработки выбора NBI
def on_nbi_pick_slider(value, plotter):
    print(f"Slider value: {value}")
    # Здесь вы можете добавить свою логику для обновления визуализации

def visualisation_with_slider(surface):
    # Инициализация plotter
    plotter = pv.Plotter()

    # Добавляем поверхность
    plotter.add_mesh(surface, color='cyan', show_edges=True, opacity=0.2)

    # Добавляем слайдер для выбора NBI
    plotter.add_slider_widget(
        callback=lambda value: on_nbi_pick_slider(value, plotter),
        rng=[1, 12],  # Диапазон NBI от 1 до 12
        title="Select NBI",
        value=1,  # Начальное значение слайдера
        pointa=(0.8, 0.9),  # Позиция на экране
        pointb=(0.98, 0.9),  # Позиция на экране
    )

    plotter.show()

if __name__ == "__main__":
    # Создайте тестовую поверхность (например, сферу) для визуализации
    sphere = pv.Sphere(radius=1.0, center=(0, 0, 0))
    
    # Запускаем визуализацию с слайдером
    visualisation_with_slider(sphere)
