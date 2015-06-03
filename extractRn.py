#   Extract-RN 
#	Input:	
#		p - Positive examples
#		U - unlabeled data
#	Output:
#		rn - Reliable negative examples set		

import numpy as np

extractRn(p, U):
	rowsU, columnsU = U.shape
	C = getPositiveLabels(p)
	n = 0.30  # porcentagem de instancias que queremos
	#rn = np.array([])

	# Um lista de entropia pra cada d em relacao a cada classe.
	# A lista tera o formato :  [entropy, instance]
	for index in xrange(0,rowsU):
		Entropy = getEntropy(C, U[index])
		myTuple = np.hstack((Entropy,U[index]))
		L = np.vstack((L,myTuple))

	ranked = getRank(L)
	rowsR, columnsR = ranked.shape

	rn = ranked[0:rows*0.3,]

	return rn




