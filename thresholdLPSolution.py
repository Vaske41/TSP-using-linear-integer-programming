from math import sqrt
from scipy.optimize import linprog
import matplotlib.pyplot as plt

inputFileName = 'input/_in263.txt'
outputFileName = 'output/_outThreshold263.txt'

EPS = 1e-11
numCities = 0
threshold = 0.65

coordinates = []

costMatrix = []

A_eq = []
b_eq = []
A_ub = []
b_ub = []
cycles = []
x = []

def dfsCycle(u, p, color: list, par: list):
    global x

    if color[u] == 2:
        return

    if color[u] == 1:
        v = []
        cur = p
        v.append(cur)

        while cur != u:
            cur = par[cur]
            v.append(cur)
        cycles.append(v)

        return

    par[u] = p

    color[u] = 1

    for city in range(numCities):
        if city == par[u] or city == u:
            continue
        val = x[encodeIndex(city, u)]
        if val >= threshold:
            dfsCycle(city, u, color, par)

    color[u] = 2

def decodeIndex(index: int) -> tuple[int, int]:
    n = 0
    while ((n + 1) * (n + 2) // 2) < index:
        n += 1
    return n, index - (n * (n + 1) // 2)

def encodeIndex(i: int, j: int) -> int:
    if i > j:
        return (i * (i + 1)) // 2 + j
    else:
        return (j * (j + 1)) // 2 + i

def countEdges(x: list[int]) -> list[int]:
    edges = [0] * numCities
    for i in range(numCities):
        for j in range(i + 1):
            if x[encodeIndex(i, j)] >= threshold:
                edges[i] += 1
                edges[j] += 1

    return edges


def distance(i: int, j: int) -> float:
    xDist = (coordinates[i][0] - coordinates[j][0]) * (coordinates[i][0] - coordinates[j][0])
    yDist = (coordinates[i][1] - coordinates[j][1]) * (coordinates[i][1] - coordinates[j][1])
    return sqrt(xDist + yDist)

def printArrayAsMatrix():
    for i in range(numCities):
        for j in range(i + 1):
            print(x[i * (i + 1) // 2 + j], end=' ')
        print('')

def totalCost(x: list[int]) -> float:
    cost = 0.0
    for index, value in enumerate(x):
        if value >= threshold:
            cost += costMatrix[index]

    return cost

def Init():
    global numCities
    global b_eq
    global inputFileName
    with open(inputFileName, 'r') as inputFile:
        for line in inputFile:
            x, y = line.split()
            x, y = float(x), float(y)
            coordinates.append((x, y))

    numCities = len(coordinates)
    b_eq = [2] * numCities

    for i in range(numCities):
        for j in range(i + 1):
            if i == j:
                costMatrix.append(0.0)
            else:
                costMatrix.append(distance(i, j))

    for i in range(numCities):
        indexList = [0] * (numCities * (numCities + 1) // 2)
        for j in range(numCities):
            if i != j:
                indexList[encodeIndex(i, j)] = 1

        A_eq.append(indexList)

def addConstrait(nodes: list[int]) -> None:
    indexList = [0] * (numCities * (numCities + 1) // 2)
    cycleNodes = set(nodes)
    otherNodes = set()
    for node in range(numCities):
        if node not in cycleNodes:
            otherNodes.add(node)
    for cycleNode in cycleNodes:
        for otherNode in otherNodes:
            indexList[encodeIndex(cycleNode, otherNode)] = -1
    A_ub.append(indexList)
    b_ub.append(-2)

def addBranch(node1: int, node2: int) -> None:
    indexList = [0] * (numCities * (numCities + 1) // 2)
    indexList[encodeIndex(node1, node2)] = 1
    A_eq.append(indexList)
    b_eq.append(1)

def resolveTwoCycles(cycles: list[list[int]]):
    firstCycle = cycles[0]
    secondCycle = cycles[1]
    bestSum = float('inf')
    edgesToAdd = [[-1, -1], [-1, -1]]
    edgesToDelete = [[-1, -1], [-1, -1]]

    for i in range(len(firstCycle)):
        firstNode1 = firstCycle[i]
        for j in range(i + 1, len(firstCycle)):
            secondNode1 = firstCycle[j]
            if x[encodeIndex(firstNode1, secondNode1)] >= threshold:
                continue
            for k in range(len(secondCycle)):
                firstNode2 = secondCycle[k]
                for l in range(k + 1, len(secondCycle)):
                    secondNode2 = secondCycle[l]
                    if x[encodeIndex(firstNode2, secondNode2)] < threshold:
                        continue
                    currentDistance = distance(firstNode1, secondNode1) + distance(firstNode2, secondNode2)
                    distance1 = distance(firstNode1, firstNode2)
                    distance2 = distance(secondNode1, secondNode2)
                    if distance1 + distance2 - currentDistance < bestSum:
                        bestSum = distance1 + distance2 - currentDistance
                        edgesToAdd[0] = [firstNode1, firstNode2]
                        edgesToAdd[1] = [secondNode1, secondNode2]
                        edgesToDelete[0] = [firstNode1, secondNode1]
                        edgesToDelete[1] = [firstNode2, secondNode2]
                    distance1 = distance(firstNode1, secondNode2)
                    distance2 = distance(secondNode1, firstNode2)
                    if distance1 + distance2 - currentDistance < bestSum:
                        bestSum = distance1 + distance2 - currentDistance
                        edgesToAdd[0] = [firstNode1, secondNode2]
                        edgesToAdd[1] = [secondNode1, firstNode2]
                        edgesToDelete[0] = [firstNode1, secondNode1]
                        edgesToDelete[1] = [firstNode2, secondNode2]

    for edges in edgesToDelete:
        x[encodeIndex(edges[0], edges[1])] = 0

    for edges in edgesToAdd:
        x[encodeIndex(edges[0], edges[1])] = 1

def visualise(x):
    for i in range(numCities):
        for j in range(numCities):
            if x[encodeIndex(i, j)] >= threshold:
                plt.plot([coordinates[i][0], coordinates[j][0]],
                         [coordinates[i][1], coordinates[j][1]], 'ro-')

    # for index, xValue in enumerate(x):
    #     if xValue >= threshold:
    #         city1, city2 = decodeIndex(index)
    #         plt.plot([coordinates[city1][0], coordinates[city2][0]],
    #                  [coordinates[city1][1], coordinates[city2][1]], 'ro-')

    for index in range(numCities):
        plt.annotate(str(index), (coordinates[index][0], coordinates[index][1]))

    plt.show()

def outputResults():
    with open(outputFileName, 'w') as outputFile:
        outputFile.write(f'Ukupna distanca: {totalCost(x)}\n')
        outputFile.write(f'Grane:\n')
        for i in range(numCities):
            for j in range(0, i):
                if x[encodeIndex(i, j)] >= threshold:
                    outputFile.write(f'{i} - {j}\n')

def setMaximumEdgeToOne(index: int, x) -> None:
    values = []
    for j in range(numCities):
        if j != index:
            values.append((x[encodeIndex(index, j)], j))

    values.sort()
    if values[-1][0] == 1:
        addBranch(index, values[-2][1])
    else:
        addBranch(index, values[-1][1])

def addConstraintsEdgesCnt(edgesCnt: list[int], x) -> None:
    for index, value in enumerate(edgesCnt):
        if value != 2:
            setMaximumEdgeToOne(index, x)

def main():
    global x

    Init()
    end = False
    while not end:
        if len(b_ub) > 0:
            result = linprog(costMatrix, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1))
        else:
            result = linprog(costMatrix, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
        x = result.x

        if not result.success:
            print('Neuspesno nadjeno resenje!')
            break

        edgesCnt = countEdges(x)
        print(f'EdgesCnt: {edgesCnt}')

        if edgesCnt.count(2) != len(edgesCnt):
            addConstraintsEdgesCnt(edgesCnt, x)
            continue

        cycles.clear()
        color = [0] * numCities
        par = [i for i in range(numCities)]
        for i in range(numCities):
            if color[i] == 0:
                dfsCycle(i, i, color, par)

        print(cycles)

        if len(cycles) == 2:
            resolveTwoCycles(cycles)
            outputResults()
            visualise(x)
            end = True
        elif len(cycles) > 2:
            for cycle in cycles:
                addConstrait(cycle)
        else:
            outputResults()
            visualise(x)
            end = True

if __name__ == '__main__':
    main()