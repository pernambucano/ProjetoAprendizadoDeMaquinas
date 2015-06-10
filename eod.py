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
import extractRn as ern
import sys

def eod(k, P, RN, U):
	T = 0.7
	C = ern.getPositiveLabels(P)
	
	for d in RN:
		#U.remove(d)
		U = deleteRow(U, d)

	print
	print '----U----'
	print U

	listOfDistances = []
	distanceTotal = 0
	for d in U:
		for p in P:
			distance = euclidianDistance(d,p)
			if distance > T:
				#U.remove(d)
				U =  deleteRow(U,d)
				break
			distanceTotal += distance
		d = np.hstack((d,[distanceTotal]))
	# get the shape of U
	Urows, Ucolumns = U.shape
	# add a last column where we can put the new labels
	Dnew = np.c_[U, np.zeros(Urows)]
	#Dnew = U # gotta test this
	PSize = P.shape[0] # number of rows
	NOutlier = 0
	
	print ''
	print '---- Dnew WITH EMPTY LABELS ----'
	print Dnew
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
	
	while flag:
		flag = False
		for di in Dnew:
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
	outlierCandidates[outlierCandidates[:,-2].argsort()]

	print outlierCandidates[:, :-3]

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
			modifiedD = d[0:-3]
			totalEntropy+= ern.getEntropy(C,modifiedD,P)

	return totalEntropy


def main():
	#print euclidianDistance(np.array([1,1,0]),np.array([2,2,0]))
	a = np.array([[1,2,3],[4,5,6],[7,8,9]])
	b = np.array([[1,2,0], [2,3,1]])
        for i in b:
            print getLabel(i)
	
	U = np.array([[1.,1.,0.],[2.,2.,0.],[6.,8.,1.],[4.,8.,1.],[3.,8.,1.],[4.,4.,1.],[1.,2.,0.],[2.,1.,0.]])
	
	normU = U / np.linalg.norm(U)
	normP = np.array([normU[0,:], normU[1,:]])
	normRN = ern.extractRn(normP, normU)
	print '------------U----------------'
	print normU
	print '------------RN----------------'
	print normRN
	#print ern.extractRn((np.array([U[0,:], U[1,:]])), U)
	#print ern.extractRn((np.array([normU[0,:], normU[1,:]])), normU)
	
	print '------------EOD----------------'
	print ''
	eod (6, normP, normRN, normU)


if __name__ == '__main__':
    main()
