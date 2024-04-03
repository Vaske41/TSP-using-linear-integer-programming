from math import sqrt
from scipy.optimize import linprog
import matplotlib.pyplot as plt

numCities = 12

coordinates = [(2.7, 33.1), (2.7, 56.8),
               (9.1, 40.3), (9.1, 52.8),
               (15.1, 49.6), (15.3, 37.8),
               (21.5, 45.8), (22.9, 32.7),
               (33.4, 60.5), (28.4, 31.7),
               (34.7, 26.4), (45.7, 25.1)]

costMatrix = []

A_eq = []
b_eq = [2] * numCities

def decodeIndex(index: int) -> tuple[int, int]:
    return index // numCities, index % numCities

def encodeIndex(i: int, j: int) -> int:
    return i * numCities + j

def countEdges(x: list[int]) -> list[int]:
    edges = [0] * numCities
    for index, val in enumerate(x):
        if val == 1:
            city1, city2 = decodeIndex(index)
            # print(city1, city2)
            edges[city1] += 1
            edges[city2] += 1
    return edges


def distance(i: int, j: int) -> float:
    xDist = (coordinates[i][0] - coordinates[j][0]) * (coordinates[i][0] - coordinates[j][0])
    yDist = (coordinates[i][1] - coordinates[j][1]) * (coordinates[i][1] - coordinates[j][1])
    return sqrt(xDist + yDist)

def printArrayAsMatrix(arr: list[int]):
    for i in range(numCities):
        print(arr[i*numCities:(i+1)*numCities])

def totalCost(x: list[int]) -> float:
    cost = 0.0
    for index, value in enumerate(x):
        row, col = decodeIndex(index)
        cost += value * distance(row, col)

    return cost

def Init():
    for i in range(numCities):
        for j in range(numCities):
            if i == j:
                costMatrix.append(0.0)
            else:
                costMatrix.append(distance(i, j))

    for i in range(numCities):
        indexList = []
        for _ in range(numCities * numCities):
            indexList.append(0)

        for index in range(numCities):
            if index != i:
                if index > i:
                    indexList[encodeIndex(index, i)] = 1
                else:
                    indexList[encodeIndex(i, index)] = 1
        A_eq.append(indexList.copy())


def main():
    Init()
    result = linprog(costMatrix, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), integrality=[1] * (numCities * numCities))
    x = result.x
    printArrayAsMatrix(x)
    cost = totalCost(x)
    print(f'Ukupna distanca: {cost}')
    printArrayAsMatrix(costMatrix)
    # print(countEdges(x))

    for index, xValue in enumerate(x):
        if xValue > 0:
            city1, city2 = decodeIndex(index)
            plt.plot([coordinates[city1][0], coordinates[city2][0]],
                     [coordinates[city1][1], coordinates[city2][1]], 'ro-')

    for index in range(numCities):
        plt.annotate(str(index), (coordinates[index][0], coordinates[index][1]))

    plt.show()

if __name__ == '__main__':
    main()



