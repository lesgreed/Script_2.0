import numpy as np

def gets(point1, point2, scale):
    """
    Генерирует массив из 'scale' точек, равномерно распределённых между point1 и point2.
    
    Parameters:
        point1 (array-like): Первая точка [x1, y1, z1].
        point2 (array-like): Вторая точка [x2, y2, z2].
        scale (int): Количество точек в результате, включая начальную и конечную.

    Returns:
        numpy.ndarray: Массив размером (scale, 3) с точками между point1 и point2.
    """
    # Создаем линейно распределенные точки между каждой координатой
    points = np.linspace(point1, point2, scale)
    return points

# Пример использования:
p1 = [0, 0, 0]
p2 = [10, 10, 10]
scale = 100

result = gets(p1, p2, scale)
print(result[0])
