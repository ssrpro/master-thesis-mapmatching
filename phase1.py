import matplotlib.pyplot as plt
from numpy import random

import math

Epsilon = 0.0000000001
Inf = 100000000
Alpha = 6
Beta = 2/3

NodeSizeThreshold = 1
NodeLevelThreshold = Inf

def epsilonRange(value, min, max):
    EpsilonThreshold = Epsilon * 10
    return value - min >= -EpsilonThreshold and value - max <= EpsilonThreshold

def main():
    points = createPoints(1000)
    tree = createBarTree(points)
    print(tree)

def createBarNode(P, R, level):
    assert checkPointsInsideRegion(P, R), "All points must lie inside the region"
    assert checkRegionAlphaBalanced(R), "Region must be alpha balanced"

    if pointSetSize(P) <= NodeSizeThreshold or level >= NodeLevelThreshold:
        return None

    cutResult = findOneCut(P, R)
    if cutResult is None:
        cutResult = findTwoCut(P, R)
    
    assert cutResult is not None, "A two-cut must always exists"

    (cutInfo, P1, R1, P2, R2) = cutResult
    nodeLeft = createBarNode(P1, R1, level+1)
    nodeRight = createBarNode(P2, R2, level+1)
    return (cutInfo, nodeLeft, nodeRight)

def createBarTree(points):
    P = makePointset(points)
    R = createRegion(P)
    root = createBarNode(P, R, 0)
    return root

def createPoints(count):
    def addTupleNoise(p):
        (x, y) = p

        ampl = Epsilon * 10000
        nx = (random.uniform(0, 1) * 2 - 1) * ampl
        ny = (random.uniform(0, 1) * 2 - 1) * ampl

        return (x + nx, y + ny)

    points = []
    for i in range(count):
        points.append((random.rand(), random.rand()))

    for i in range(len(points)):
        points[i] = addTupleNoise(points[i])

    return points

def makePointset(points):
    pointsConverted = [(x, y, x - y) for (x, y) in points]

    pointsSortedX = pointsConverted.copy()
    pointsSortedY = pointsConverted.copy()
    pointsSortedZ = pointsConverted.copy()

    pointsSortedX.sort(key=lambda x: x[0])
    pointsSortedY.sort(key=lambda x: x[1])
    pointsSortedZ.sort(key=lambda x: x[2])

    P = (pointsSortedX, pointsSortedY, pointsSortedZ)
    return P

def createRegionFromPoints(points):
    return [
        min(x for (x, y, z) in points),
        max(x for (x, y, z) in points),
        min(y for (x, y, z) in points),
        max(y for (x, y, z) in points),
        min(z for (x, y, z) in points),
        max(z for (x, y, z) in points)
    ]

def createRegion(P):
    (pointsSortedX, pointsSortedY, pointsSortedZ) = P

    xmin = pointsSortedX[0][0]
    xmax = pointsSortedX[-1][0]
    ymin = pointsSortedY[0][1]
    ymax = pointsSortedY[-1][1]
    zmin = pointsSortedZ[0][2]
    zmax = pointsSortedZ[-1][2]

    R = (xmin, xmax, ymin, ymax, zmin, zmax)
    return R

def pointInR(p, R):
    (x, y, z) = p
    (xmin, xmax, ymin, ymax, zmin, zmax) = R

    return epsilonRange(x, xmin, xmax) and epsilonRange(y, ymin, ymax) and epsilonRange(z, zmin, zmax)

def checkPointsInsideRegion(P, R):
    (pointsSortedX, _, _) = P

    for p in pointsSortedX:
        if not pointInR(p, R):
            return False
        
    return True

def checkRegionAlphaBalanced(R):
    (xmin, xmax, ymin, ymax, zmin, zmax) = R

    diamx = xmax - xmin
    diamy = ymax - ymin
    diamz = 0.5 * (zmax - zmin)

    ar = max(diamx, diamy, diamz) / min(diamx, diamy, diamz)
    return ar <= Alpha

def pointSetSize(P):
    (pointsSortedX, _, _) = P
    return len(pointsSortedX)

def getAllOuterVerticesR(R):
    (xmin, xmax, ymin, ymax, zmin, zmax) = R
    allIntersections = []

    allIntersections.append((xmin, ymin, xmin - ymin)) # (x-, y-)
    allIntersections.append((xmin, ymax, xmin - ymax)) # (x-, y+)
    allIntersections.append((xmin, xmin - zmin, zmin)) # (x-, z-)
    allIntersections.append((xmin, xmin - zmax, zmax)) # (x-, z+)
    
    allIntersections.append((xmax, ymin, xmax - ymin)) # (x+, y-)
    allIntersections.append((xmax, ymax, xmax - ymax)) # (x+, y+)
    allIntersections.append((xmax, xmax - zmin, zmin)) # (x+, z-)
    allIntersections.append((xmax, xmax - zmax, zmax)) # (x+, z+)

    allIntersections.append((ymin + zmin, ymin, zmin)) # (y-, z-)
    allIntersections.append((ymax + zmin, ymax, zmin)) # (y+, z-)
    allIntersections.append((ymin + zmax, ymin, zmax)) # (y-, z+)
    allIntersections.append((ymax + zmax, ymax, zmax)) # (y+, z+)

    return [p in allIntersections if pointInR(p, R)]

def getLineIntersectingR(lineAxis, lineValue, R):
    (xmin, xmax, ymin, ymax, zmin, zmax) = R
    allIntersections = []

    if lineAxis == 0: # x-axis
        x = lineValue
        allIntersections.append((x, ymin, x - ymin)) # (X, y-)
        allIntersections.append((x, ymax, x - ymax)) # (X, y+)
        allIntersections.append((x, x - zmin, zmin)) # (X, z-)
        allIntersections.append((x, x - zmax, zmax)) # (X, z+)

    elif lineAxis == 1: # y-axis
        y = lineValue
        allIntersections.append((xmin, y, xmin - ymin)) # (Y, x-)
        allIntersections.append((xmax, y, xmax - ymax)) # (Y, x+)
        allIntersections.append((y + zmin, y, zmin)) # (Y, z-)
        allIntersections.append((y + zmax, y, zmax)) # (Y, z+)

    elif lineAxis == 2: # z-axis
        z = lineValue
        allIntersections.append((xmin, xmin - z, z)) # (Z, x-)
        allIntersections.append((xmax, xmax - z, z)) # (Z, x+)
        allIntersections.append((ymin + z, ymin, z)) # (Z, y-)
        allIntersections.append((ymax + z, ymax, z)) # (Z, y+)

    return [p for p in allIntersections if pointInR(p, R)]

def getCutResult(R, P, cutType, cutAxis, cutValue):
    cutInfo = (cutType, cutAxis, cutValue)

    P1 = createNewP()
    P2 = createNewP()
    for i in [0, 1, 2]: # ["x", "y", "z"]
        for p in P[i]:
            if p[cutAxis] < cutValue:
                P1[i].append(p)
            else:
                P2[i].append(p)

    V = getOuterVerticesR(R)
    I = getLineIntersectingR(cutAxis, cutValue)
    R1 = createRegionFromPoints([p for p in V if p[cutAxis] < cutValue] + I)
    R2 = createRegionFromPoints([p for p in V if p[cutAxis] > cutValue] + I)

    return (cutInfo, P1, R1, P2, R2)

def findOneCut(R, P):
    RidCollection = findRi(R)

    for i in [0, 1, 2]: # ["x", "y", "z"]
        (Ril, Rir) = RidCollection[i]

        if Ril is not None and Rir is not None:
            n = countP(P)
    
            for (p_il, p_ir) in zip(P[i][:-1], P[i][1:]): # Iterate over each neighbouring pair
                if min(p_il[i], Ril) <= max(p_ir[i], Rir):
                    cutValue = 1/2 * (min(p_il[i], Ril) + max(p_ir[i], Rir))
                    return getCutResult(R, P, cutType="one", cutAxis=i, cutValue=cutValue)

    return None # There does not exist a one-cut

def findTwoCut(P, R):
    pass

main()