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


def eod(k, P, RN, U):
	T = 0.7

	for d in RN:
		#U.remove(d)
		U = deleteRow(U, d)

	for d in U:
		for p in P:
			distance = euclidianDistance(d,p)

			if distance > T:
				#U.remove(d)
				U =  deleteRow(U,d)
				break
	# get the shape of U
	Urows, Ucolumns = U.shape
	# add a last column where we can put the new labels
	Dnew = np.c_[U, np.zeros(Urows)]
	#Dnew = U # gotta test this
	PSize = P.shape[0] # number of rows
	NOutlier = 0

	for d in Dnew:
		NOutlier = NOutlier + 1

		if NOutlier <= k-PSize:
			#label d as O // Outlier
			d = putLabel(d, 'O')
		else:
			#label d as N // Non-outlier
			d = putLabel(d, 'N')

	flag = True

	while flag:
		for di in Dnew:
			if getLabel(di) == 'N':
				for dj in Dnew:
					if getLabel(dj) == 'O':
						#changing labels
						temp = dj[-1]
						dj[-1] = di[-1]
						di[-1] = temp
						
						#calculate the new entropy

				#if maximum decrease of entropy achieved
					#swap the label of di and dj with minimum entropy value

		#if entropy has not changed
			#flag = false
	kOutputs = []
	#select k-PSize with label O
	#rank k-PSize instances with label O (rank based on the euclidianDistance calculated

	return kOutputs


def deleteRow(Array, row):
    sliced = np.where(np.all(Array==row, axis=1))
    Array = np.delete(Array, sliced, 0)
    return Array

def exchangeLabels(di, dj):



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



def main():
	#print euclidianDistance(np.array([1,1,0]),np.array([2,2,0]))
	a = np.array([[1,2,3],[4,5,6],[7,8,9]])
	b = np.array([[1,2,0], [2,3,1]])
        for i in b:
            print getLabel(i)
if __name__ == '__main__':
    main()
