#   EOD (Entropy-base Outlier Detection)
#	Input:	
#		k - number of outliers
#		P - positive examples
#		RN - negative samples
#		U - unlabeled dataset
#	Output:
#		op - k ranked outlier point

import numpy as np


def eod(k, P, RN, U):
	T = 0.7
	
	for d in RN:
		U.remove(d)
	
	for d in U:
		for p in P:
			distance = euclidianDistance(d,p)
			
			if distance > T:
				U.remove(d)
				break
				
	Dnew = U
	PSize = len(p)
	NOutlier = 0
	
	for d in Dnew:
		NOutlier = NOutlier + 1

		if NOutlier <= k-PSize:
			pass
			#label d as O // Outlier
		else:
			pass
			#label d as N // Non-outlier

	flag = true

	while flag:
		for di in Dnew:
			if getLabel(di) == 'N':
				for dj in Dnew:
					if getLabel(dj) == 'O':
						pass
					#exchange the label of N with O and calculate the new entropy

				#if maximum decrease of entropy achieved
					#swap the label of di and dj with minimum entropy value

		#if entropy has not changed
			#flag = false
	kOutputs = []
	#select k-PSize with label O
	#rank k-PSize instances with label O (rank based on the euclidianDistance calculated

	return kOutputs

def euclidianDistance(d,p):
	return np.linalg.norm(d-p)
	
def getLabel(d):
	pass
	
def main():
	print euclidianDistance(np.array([1,1]),np.array([2,2]))
	
if __name__ == '__main__':
    main()