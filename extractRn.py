#   Extract-RN 
#	Input:	
#		p - Positive examples
#		U - unlabeled data
#	Output:
#		rn - Reliable negative examples set		

import numpy as np

def extractRn(P, U):
	rowsU, columnsU = U.shape
	C = getPositiveLabels(P)
	n = 2  # numero de instancias que queremos ## Possivel mudanca pra porcentagem
	L = np.array(columnsU + 1) # lista de exemplos + a entropia
	
	Entropy = []
	for d in U:
		Entropy.append(getEntropy(C,d,P))

	rn = getRank(Entropy, U, n)

	# rn e uma lista de python
	# se for necessario transformar em um numpy.array:
	#	rn = np.array(rn)
	return rn

def getPositiveLabels(P):
	
	#para cada elemento em p, e conjunto l vazio
		#	se a classe de p nao esta em l,
		#	adicionar classe de p em l
		# *** considera que a classe e a ultima posicao ***
	labels = []
	
	for instance in P:
		if instance[-1] not in labels:
			labels.append(instance[-1])

	return labels 

#   getEntropy
#	Input:	
#		C - class labels
#		d - current instance
#		P - positive instances 

#	Output:
#		rn - Reliable negative examples set		
def getEntropy(C,d,P):

	entropy = 0
	
	for classe in C:
		probability = getProbability(C,d,P,classe)
		if(probability != 0):
			entropy += probability * np.log2(probability)
	
	return entropy
	
	
def getProbability(C,d,P,classe):

	pj = getCenterOfClass(P,classe)
	distance1 = euclidianDistance (d,pj)
	
	distance2 = 0
	for classe in C:
		for instance in P:
			if(instance[-1] == classe):
				# if the class has at least one item
				pj = getCenterOfClass(P,classe)
				distance2 += euclidianDistance (d,pj)
				break
	
	# probability = distance1/distance2 - when we have just one class, this always returns 1
	
	#ESSA PARTE TA CONFUSA NO ARTIGO, PRECISA SER REFEITA (CALCULO COM A NORMALIZACAO)
	if(distance1 == distance2): 			#<--- BUG AQUI, TEM QUE ARRUMAR
		probability = distance1;			#<--- BUG AQUI, TEM QUE ARRUMAR
	else:						 			#<--- BUG AQUI, TEM QUE ARRUMAR
		probability = distance1/distance2 	#<--- BUG AQUI, TEM QUE ARRUMAR
		
	return probability

def getCenterOfClass(P, classe):
	columns = len(P[0])
	count = 0
	#columns-1 because the last column is the class
	center = np.zeros(shape=(1,columns-1))
	
	#this part checks all the instances in P that have the class == classe and
	#sums the vectors to find the mean value (the center)
	for instance in P:
		if instance[-1] == classe:
			#newInstance removes the class from the calculation
			newInstance = instance[0:columns-1]
			center = np.sum([center, newInstance], axis=0)
			count+= 1
	
	#finds the center and add the class again
	newCenter = [x / count for x in center] # can divide by 0!
	newCenter = np.append(newCenter, np.array(classe))
	return newCenter

	
def euclidianDistance(d,p):
	#removes the last column, which is the class column, the calculates the euclidian distance
	columns = len(d)
	newD = d[0:columns-1]
	newP = p[0:columns-1]

	return np.linalg.norm(newD-newP)
	
# Recebe uma lista de entropias e retorna uma lista com os n instancias com maiores entropias
def getRank(Entropy, U, n):

	#finds the increasing order of entropies
	index = np.argsort(Entropy)
	
	# se for usar porcentagem :
	# 	index = np.argsort(Entropy)[::-1]
	# 	indices = index[0:U.shape[0]*n]

	instances = []
	for i in reversed(index): # reversed because we want the highest values
		instances.append(U[i])

	# just return the n first elements
	return instances[:n]

def main():
	#print np.array([[1,1,0],[2,2,0]])
	#print getCenterOfClass((np.array([[1.,1.,0.],[2.,2.,0.]])), 0)
	#print getProbability([0,1], np.array([2.0,2.5,0.]),(np.array([[1.,1.,0.],[2.,2.,0.]])), 0)
	print extractRn((np.array([[1.,1.,0.],[2.,2.,0.]])), (np.array([[3.,8.,1.],[1.,1.,0.],[2.,2.,0.], [4.,4.,1.]])))
	
if __name__ == '__main__':
    main()