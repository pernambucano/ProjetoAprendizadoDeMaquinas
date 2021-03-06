#   EOD (Entropy-base Outlier Detection)
#	Input:
#		k - number of outliers - integer
#		P - positive examples - numpy.array
#		RN - negative samples - numpy.array
#		U - unlabeled dataset - numpy.array
#	Output:
#		op - k ranked outlier point - numpy.array
#

import numpy as np
import newExtractRN as ern
import sys
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt


def eod(k, P, RN, U):
	T = 0.7
	C = ern.getPositiveLabels(P)
	print '---U---'
	print U.shape

        counter = 0
	for d in RN:
	        if d[-1] == 4:
	            counter +=1
		#U.remove(d)
                U = deleteRow(U, d)

        print 'positive deleted', counter
	##printGraph(U, 'U after RN')
	print
	print '----U After RN----'
	print U.shape

	# It will delete d if only if d is far from every outlier, i.e. distance(d, All Outliers) > T
	for d in U:
		isFar = True
		minimumDistance = 10000
		for p in P:
		        # if p != d
			distance = euclidianDistance(d, p)
			if distance < minimumDistance:
			    minimumDistance = distance

		if minimumDistance > T:
			#U.remove(d)
                        #print 'removing ', d
                    #   if d[-1] != 4:
            			U =  deleteRow(U,d)

		#d = np.hstack((d,[distanceTotal]))

        print 'U after distance', U.shape
	#printGraph(U, 'U after distance')
	# get the shape of U
	Urows, Ucolumns = U.shape
	# add a last column where we can put the new labels
	Dnew = np.c_[U, np.zeros(Urows)]
	#Dnew = U # gotta test this
	PSize = P.shape[0] # number of rows
	NOutlier = 0

	# print ''
	# print '---- Dnew WITH EMPTY LABELS ----'
	# print Dnew
	for d in Dnew:
		NOutlier = NOutlier + 1

		if NOutlier <= k-PSize:
			#label d as O // Outlier
			d = putLabel(d, 'O')
		else:
			#label d as N // Non-outlier
			d = putLabel(d, 'N')

	flag = True

	globalEntropy = sys.maxint

	# while flag:
	# 	flag = False
	# 	for di in Dnew:
	# 		if getLabel(di) == 'N':
	# 			for dj in Dnew:
	#
	# 				if getLabel(dj) == 'O':
	# 					#changing labels
	# 					exchangeLabels(di, dj)
	#
	# 					currentEntropy = getSetEntropyForLabel(C, P, Dnew, 'O')
	#
	# 					if(currentEntropy < globalEntropy):
	# 						globalEntropy = currentEntropy
	# 						flag = True
	# 					else:
	# 						#if isn't better, undo
	# 						exchangeLabels(di, dj)

	while flag:
		flag = False
		for di in Dnew:
			maximumDecreaseAchieved = False
			if getLabel(di) == 'N':
				for dj in Dnew:

					if getLabel(dj) == 'O':
						#changing labels
						exchangeLabels(di, dj)

						currentEntropy = getSetEntropyForLabel(C, P, Dnew, 'O')

						if(currentEntropy < globalEntropy):
							globalEntropy = currentEntropy
							flag = True
						else:
							#if isn't better, undo
							exchangeLabels(di, dj)

	outlierCandidates = np.array([]).reshape(0,Ucolumns+1)

	print ''
	print '------DNEW AFTER LABELS---------'
	print Dnew
	for d in Dnew:
		if getLabel(d) == 'O':
			outlierCandidates = np.vstack([outlierCandidates, d])

	#select k-PSize with label O
	#rank k-PSize instances with label O (rank based on the euclidianDistance calculated
	#TODO: FALTA ORDENAR PELA DISTANCIA
	print ''
	print '------OUTLIER CANDIDATES---------'
	# TODO: delete the label and distance columns
	#print outlierCandidates

	# ranking using the -2 column
	#outlierCandidates[outlierCandidates[:,-2].argsort()]

	print outlierCandidates[:, :]

	# print 'calculatedPrecision', calculatePrecision(outlierCandidates, 2)
	return outlierCandidates


def deleteRow(Array, row):
    sliced = np.where(np.all(Array==row, axis=1))
    Array = np.delete(Array, sliced, 0)
    return Array

def exchangeLabels(di, dj):
		temp = dj[-1]
		dj[-1] = di[-1]
		di[-1] = temp

def printGraph(T, title):
        sliced = np.where(np.all(T[-1]==2, axis=0))
        print sliced
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(T[:,0], T[:,1], T[:,2],
	        '^', markersize=8, alpha=0.5, color='red', label='class2')

	plt.title(title)
	ax.legend(loc='upper right')

	plt.show()

def euclidianDistance(d,p):
	#removes the last column, which is the class column, the calculates the euclidian distance
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

	print 'db1'
	print database1[:5,:]
	#normData = database1[:,:-1]
	#normData = normData / np.linalg.norm(normData)
	#normData = (normData - normData.min(axis=0)) / normData.ptp(axis=0)
	#normData = np.hstack([normData, database1[:,-1:]])
	normData = normalize(database1)

	print 'norm'
	print normData[:5,:]
	benign = np.array([]).reshape(0,columns)
	malign = np.array([]).reshape(0,columns)

	for d in normData:
		if d[-1] == 2:
			benign = np.vstack([benign, d])
		else:
			malign = np.vstack([malign, d])

	np.random.shuffle(malign)


	reducedMalign = malign [:10,:]

    	print 'benign ', benign.shape
	P = reducedMalign[:3,:]
    	benign = benign[:357,:]
	U = np.vstack([benign,reducedMalign])
	np.random.shuffle(U)

	print ' --- P ---'
	print P
	print '--- U ---'
	print U
	return P, U

def calculatePrecision(outlierCandidates, nonOutlierClass):
	rows, columns = outlierCandidates.shape
	nonOutlier = 0.0
	for d in outlierCandidates:
		#TODO: ESTOU CONSIDERANDO QUE A CLASSE E A PENULTIMA POSICAO - JA QUE TEM O LABEL
		if (d[-2] == nonOutlierClass):
			nonOutlier+= 1

	return (1.0- (nonOutlier/rows))

def normalize(T):
	norm =   (T[:, :-1] - T[:, :-1].min(axis=0)) / T[:,:-1].ptp(axis=0)
        print 'norm here', norm[:3]
	norm = np.hstack([norm, T[:, -1:]])
        print norm[:3]
	return norm
def main():
	#print euclidianDistance(np.array([1,1,0]),np.array([2,2,0]))
	#a = np.array([[1,2,3],[4,5,6],[7,8,9]])
	#b = np.array([[1,2,0], [2,3,1]])
    #    for i in b:
    #        print getLabel(i)

	#U = np.array([[1.,1.,0.],[2.,2.,0.],[6.,8.,1.],[4.,8.,1.],[3.,8.,1.],[4.,4.,1.],[1.,2.,0.],[2.,1.,0.]])

	# normU = U / np.linalg.norm(U)
	# normP = np.array([normU[0,:], normU[1,:]])
	#normRN = ern.extractRn(normP, normU)
	#print '------------U----------------'
	#print normU
	#print '------------RN----------------'
	#print normRN
	#print ern.extractRn((np.array([U[0,:], U[1,:]])), U)
	#print ern.extractRn((np.array([normU[0,:], normU[1,:]])), normU)
	#getBreastCancerData()

	# print '------------EOD----------------'
	# print ''	#eod (6, normP, normRN, normU)


	### Test Paulo
	## Class1 = non-outliers
	## Class2 = outliers
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
	#norm_class2 = np.array([[0.9, 0.9, 0.9, 1], [0.91, 0.91, 0.91, 1], [0.92, 0.92, 0.92, 1], [0.01, 0.01, 0.01, 1], [0.02, 0.02, 0.02,1] , [0.03,0.03,0.03, 1], [0.04,0.04,0.04,1],[0.94,0.94,0.94,1], [0.05,0.05,0.05,1], [1,1,1,1]])
	norm_class2 = normalize(class2_sample)
	np.random.shuffle(norm_class2)

	P = norm_class2[:3,:]
	#print 'P', P
	#U = np.vstack([norm_class1, norm_class2])
	#np.random.shuffle(U)
	##printGraph(U, 'U before anything')
	# normRN = ern.extractRn(normP,normU)
	# fig = plt.figure(figsize=(8,8))
	# ax = fig.add_subplot(111, projection='3d')
	# ax.plot(class2_sample[:,0], class2_sample[:,1], class2_sample[:,2],
	#         '^', markersize=8, alpha=0.5, color='red', label='class2')
    #     ax.plot(class1_sample[:,0], class1_sample[:,1], class1_sample[:,2],
    #     '^', markersize=8, alpha=0.5, color='blue', label='class1')
	#
	#
	# ax.legend(loc='upper right')
	#
	# plt.show()
	#P , U = getBreastCancerData()

	# result = eod (10, normP, normRN, normU)
	P, U = getBreastCancerData()
	#U = np.vstack([norm_class1, norm_class2])
	np.random.shuffle(U)
	RN = ern.newExtractRN(P,U)
	result = eod (13, P, RN, U)

	#printGraph(result, 'Final result')

	#print P
	#print RN



if __name__ == '__main__':
    main()
