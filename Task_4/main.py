import numpy as np
import matplotlib.pyplot as plt
import random as rnd

SIZE_COORDS = 10


def generation_points(N, M):
    points = []
    for i in range(N):
        point = []
        for j in range(M):
            point.append(rnd.randint(-SIZE_COORDS, SIZE_COORDS))
        points.append(np.array(point))
    return points


def p1_is_dominant(point1, point2):
    return (point1 - point2).min() >= 0


def get_pareto_front(N, M, points):
    pareto_front = []
    mark_points = np.zeros(N) # Точно не оптимальные по Парето
    for i in range(N):
        if mark_points[i] != -1:
            is_optimal = True
            for j in range(N):
                if mark_points[j] not in [-1, 1] and i != j:
                    if p1_is_dominant(points[j], points[i]):
                        is_optimal = False
                        break
                    if p1_is_dominant(points[i], points[j]):
                        mark_points[j] = -1 # Т.е. эту точку в дальнейшем нет смысла проверять на оптимальность
            if is_optimal:
                pareto_front.append(points[i])
                mark_points[i] = 1
                # Надо понимать, что все, кто на этом этапе не помечены -1, не доминируют и не одминируются точкой i
            else:
                mark_points[i] = -1
    return pareto_front


def show_points_line(M, points, pareto_front):
    plt.title("Отображение Парето фронт")
    plt.xlabel("Номер координаты точки")
    plt.ylabel("Значение координаты точки")
    plt.grid()

    coords = np.arange(M)
    for point in points:
        plt.plot(coords, point, 'D--')
    for pareto_point in pareto_front:
        plt.plot(coords, pareto_point, 'D-')

    plt.show()


def main():
    N = int(input('N (count of points): '))
    M = int(input('M (count of coords): '))
    points = generation_points(N, M)
    # for point in points:
    #     print(point)
    # print('------------------------------------')
    pareto_front = get_pareto_front(N, M, points)
    # for point in pareto_front:
    #     print(point)
    show_points_line(M, points, pareto_front)


if __name__ == "__main__":
    main()

