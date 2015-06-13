import numpy as np

def newExtractRN(P, U):
    positiveLabels = getPositiveLabels(P)
    numInstances = 0.3

    entropyList = []
    for di in U:
        entropyList.append(getEntropy(positiveLabels, di, P))

    reliableNegative = getRank(entropyList, U, numInstances)
    return reliableNegative

def getPositiveLabels(P):
    labels = []
    for p in P:
        if p[-1] not in labels:
            labels.append(p[-1])

    return labels

def getEntropy(positiveLabels, di, P):
    entropy = 0
    probability = 0
    for classLabel in positiveLabels:
        probability = getProbability(positiveLabels, di, P, classLabel)
        if probability != 0:
            entropy += probability * np.log(probability)

    return entropy

def getProbability(positiveLabels, di, P, classLabel):
    distance_numerator = 1000
    for pj in P:
        dist_temp = euclidianDistance(di, pj)
        if dist_temp < distance_numerator:
            distance_numerator = dist_temp

    distance_denominator = 0
    for classLabel in positiveLabels:
        for p in P:
            dist_temp = euclidianDistance(di, pj)
            distance_denominator += dist_temp

    probability = distance_numerator/distance_denominator
    return probability

def euclidianDistance(d, p):
    numOfcolumns = len(d)
    new_d = d[0:numOfcolumns - 1]
    new_p = p[0:numOfcolumns - 1]

    return np.linalg.norm(new_d - new_p)


def getRank(Entropy, U, numInstances):
    indices = np.argsort(Entropy)
    instances = []

    for index in reversed(indices):
        instances.append(U[index])

    numOfRows = len(instances)
    return instances[0:int(numOfRows*numInstances)]
