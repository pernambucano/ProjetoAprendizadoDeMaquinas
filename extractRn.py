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
	n = 30  # numero de instancias que queremos ## Possivel mudanca pra porcentagem
	L = np.array(columnsU + 1) # lista de exemplos + a entropia
	
	Entropy = []
	for d in U:
		Entropy.append(getEntropy(C,d))

	rn = getRank(Entropy, U, n)

	# rn e uma lista de python
	# se for necessario transformar em um numpy.array:
	#	rn = np.array(rn)
	return rn

getPositiveLabels(p):
	pass 

getEntropy(C,d):
	pass

# Recebe uma lista de entropias e retorna uma lista com os n instancias com maiores entropias
getRank(Entropy, U, n):
	index = np.argsort(Entropy)[::-1][n] # testar
	# se for usar porcentagem :
	# 	index = np.argsort(Entropy)[::-1]
	# 	indices = index[0:U.shape[0]*n]

	instances = []
	for i in index:
		instances.append(U[i])

	return instances