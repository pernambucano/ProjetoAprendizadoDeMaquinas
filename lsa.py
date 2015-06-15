# Escolhe os k primeiros vetores como outliers
# o resto sera non-outlier
# 	cria dict com as frequencias de cada atributo so dos non-outliers
#
# pra cada non-outlier
#	pra cada outlier
#		troca de label com o outlier
#		testa a entropia do conjunto de non-outliers
#		se entropia minima foi atingida:
#			troca de label permanentemente 
#			atualiza dict

import numpy as np

def lsa(D, k):
	counter = 0
	dRows, dColumns = D.shape
	labelIndices = np.zeros(dRows)
	
	# Phase 1-initialization
	for index in xrange(dRows):
		counter += 1
		if counter <= k:
			#label t as an outlier with flag "1
			labelIndices[index] = 1

			
	# at this point, D[k+1:] is compound of non-outliers
	# and we need to create m  dicts, one for each attribute with their values and frequences
	newD = D[k:]
	attributesDict = {}
	for columns in xrange(newD.shape[1]):
		attributesDict[columns] = createFrequencyDict(columns, newD)
#	print attributesDict
	
# Phase 2- Iteration
# 	not_moved = True
# 	while not_moved:
# 		for d in D:
# 			if getLabel(d) == 0:
# 				#foreach record o in current k outliers:
# 				#	exchanging the label of t with that of o and evaluating the change of entropy
# 				#if maximal decrease on entropy is achieved by record b:
# 				#	swap the labels of t and b
# 				#	update hash tables using t and b
# 				# 	not_moved = false
# 				
# 	#return outliers						 	
	

def createFrequencyDict(a, U):

	values = {}

	for index, element in enumerate(U):
		value = element[a]

		if value in values:
			frequency = values[value] + 1
			values.update({value: frequency})
		else:
			values[value] = 1
	
	return values
	
def main():
	D = np.array([['a', 'e', 'm'], ['a', 'd', 'n'], ['b', 'g', 'm'], ['c', 'd', 'n'], ['c', 'g', 'm'], ['c', 'f', 'n']])
	lsa(D, 2)

if __name__ == '__main__':
	main()