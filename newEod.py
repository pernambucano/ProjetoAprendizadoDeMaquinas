import numpy as np
import newExtractRN as ern
import sys


def newEod(k, P, RN, U):
    T = 0.7
    positiveLabels = ern.getPositiveLabels(P)

    counter = 0
    for di in RN:
        if di[-1] == 4:
            counter += 1
        U = deleteRow(U, di)

    print 'foram deletados', counter, ' exemplos positivos'
    for di in U:
        minimumDistance = 1000
        for p in P:
            distance = euclidianDistance(di, p)
            if distance < minimumDistance:
                minimumDistance = distance

        if minimumDistance > T:
            U = deleteRow(U,di)


    numRowsU, numColumnsU = U.shape

    dNew = np.c_[U, np.zeros(numRowsU)]

    sizeP = P.shape[0]
    numOutliers = 0
    for di in dNew:
        numOutliers += 1

        if numOutliers <= k - sizeP:
            di = putLabel(di, 'O')
        else:
            di = putLabel(di, 'N')


    flag = True
    globalEntropy = sys.maxint

    while flag:
        flag = False
        for di in dNew:
            if getLabel(di) == 'N':
                for dj in dNew:
                    if getLabel(dj) == 'O':
                        exchangeLabels(di, dj)
                        currentEntropy = getSetEntropyForLabel(positiveLabels, P, dNew, 'O')

                        if currentEntropy < globalEntropy :
                            globalEntropy = currentEntropy
                            flag = True
                        else:
                            exchangeLabels(di, dj)

    outlierCandidates = np.array([]).reshape(0, numColumnsU+1)

    for di in dNew:
        if getLabel(di) == 'O':
            outlierCandidates = np.vstack([outlierCandidates, di])

    print outlierCandidates
    return outlierCandidates

def deleteRow(Array, row):
    sliced = np.where(np.all(Array==row, axis=1))
    Array = np.delete(Array, sliced, 0)
    return Array

def exchangeLabels(di, dj):
		temp = dj[-1]
		dj[-1] = di[-1]
		di[-1] = temp

def euclidianDistance(d,p):
	columns = len(d)
	newD = d[0:columns-1]
	newP = p[0:columns-1]
	return np.linalg.norm(newD-newP)

def getLabel(d):

    if d[-1] == 0:
        return 'N'
    else:
        return 'O'

# If N, then it will add 0 as the last column, if O it will add 1
def putLabel(d, label):

    if label == 'N':
        d[-1] = 0
    else:
        d[-1] = 1

    return d

def getSetEntropyForLabel(C, P, Dnew, label):

	totalEntropy = 0.0
	for d in Dnew:
		if getLabel(d) == label:
			#removing the label
			modifiedD = d[0:-2]
			totalEntropy+= ern.getEntropy(C,modifiedD,P)

	return totalEntropy

def getBreastCancerData():
	database1 = np.loadtxt('data/breast-cancer-wisconsin.data',delimiter=',')
	database1 = database1[:, 1:]
	(rows, columns) = database1.shape

        normData = database1
	#normData = normalize(database1)

	benign = np.array([]).reshape(0,columns)
	malign = np.array([]).reshape(0,columns)

	for d in normData:
		if d[-1] == 2:
			benign = np.vstack([benign, d])
		else:
			malign = np.vstack([malign, d])

	np.random.shuffle(malign)
	reducedMalign = malign [:10,:]

        benign = benign[0:357,: ]
	P = reducedMalign[:3,:]
	U = np.vstack([benign,reducedMalign])
	np.random.shuffle(U)

	return P, U


def testData():
    	mu_vec1 = np.array([0,0,0])
    	cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    	class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 300).T
    	class1_sample = class1_sample.T
    	class1_rows, class1_columns = class1_sample.shape
    	class1_sample = np.c_[class1_sample, np.zeros(class1_rows)]

    	mu_vec2 = np.array([1,1,1])
    	cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    	class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 10).T
    	class2_sample =  class2_sample.T
    	class2_rows, class2_columns = class2_sample.shape
    	class2_sample = np.c_[class2_sample, np.ones(class2_rows)]
    	norm_class1 = normalize(class1_sample)
    	norm_class2 = normalize(class2_sample)
    	np.random.shuffle(norm_class2)

    	P = norm_class2[:3,:]
    	U = np.vstack([norm_class1, norm_class2])

        return P, U


def normalize(T):
	norm =   (T[:, :-1] - T[:, :-1].min(axis=0)) / T[:,:-1].ptp(axis=0)
	norm = np.hstack([norm, T[:, -1:]])
	return norm


def main():
    P, U = getBreastCancerData()
    #P, U = testData()
    np.random.shuffle(U)
    RN = ern.newExtractRN(P,U)
    result = newEod(10, P, RN, U)

if __name__ == '__main__':
    main()
