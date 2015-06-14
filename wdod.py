from __future__ import division
import numpy as np


def UInd(a, U):

	values = {}
	for index, element in enumerate(U):
		value = element[a]
		if value in values:
			list = values[value]
			list.append(index)
			values.update({value: list})
		else:
			values[value] = [index]
	
	return values
	
def E(a, U):

	values = UInd(a,U)
	
	rows, columns = U.shape
	moduleU = rows
	
	entropy = 0
	
	for key, value in values.iteritems():	
		entropy += (len(value)/moduleU) * (1-(len(value)/moduleU))
		
	return entropy

def EOtimizado(ind, U):

	values = ind
	
	rows, columns = U.shape
	moduleU = rows
	
	entropy = 0
	
	for key, value in values.iteritems():	
		entropy += (len(value)/moduleU) * (1-(len(value)/moduleU))
		
	return entropy
	
def W(a,U,A):

	denominador = 1 - E(a,U)
	divisor = 0
		
	for attr in A:
		divisor += 1-E(attr,U)

	return denominador/divisor
	
def WOtimizado(a,U,A,entropyList):

	denominador = 1 - entropyList[a]

	divisor = 0
		
	for attr in A:
		divisor += 1 - entropyList[attr]

	return denominador/divisor

def ADens(x,a,U):

	elementsWithXValue = UInd(a, U)[x[a]]
	
	rows, columns = U.shape
	moduleU = rows
	density = len(elementsWithXValue)/moduleU
	
	return density
	
def ADensOtimizado(x,a,U,indList):

	elementsWithXValue = indList[a][x[a]]
	
	rows, columns = U.shape
	moduleU = rows
	density = len(elementsWithXValue)/moduleU
	
	return density

def WDens(x,U,A):

	density = 0
	
	for attr in A:
		density += ADens(x,attr,U) * W(attr,U,A)
		
	return density
	
def WDensOtimizado(x,U,A,indList,entropyList):

	density = 0
	for attr in A:
		density += ADensOtimizado(x,attr,U,indList) * WOtimizado(attr,U,A,entropyList)
		
	return density
	
def WDOD(U,A,threshold):

	rows, columns = U.shape
	outliers = np.zeros(shape=(0,columns))
	
	for x in U:
		density = WDens(x,U,A)
		
		if density < threshold:
			outliers = np.vstack([outliers, x])
	
	print outliers
	
def WDODOtimizado(U,A,threshold):

	rows, columns = U.shape
	outliers = np.zeros(shape=(0,columns))
	
	indList = []
	entropyList = []
	
	for attr in A:
		ind = UInd(attr,U)
		indList.append(ind)
		entropyList.append(EOtimizado(ind,U))
		
	for x in U:
		
		density = WDensOtimizado(x,U,A,indList,entropyList)
		if density < threshold:
			outliers = np.vstack([outliers, x])
	
	print outliers


def getBreastCancerData():
	database1 = np.loadtxt('data/breast-cancer-wisconsin.data',delimiter=',')
	database1 = database1[:, 1:]

	return database1
	
def getLymphographyData():
	U = np.loadtxt('data/lymphography.all',delimiter=',')
	U = U[:,:] # removing the class label

	rows,columns = U.shape
	A = range(0,columns)
	return U,A

def getTestData():
	U = np.array([["A","E","M"],["A","D","N"],["B","G","M"],["C","D","N"],["C","G","M"],["C","F","N"]])
	rows,columns = U.shape
	A = range(0,columns)
	
	return U,A
	
def paperTest(U,A):
	print '---UInd---'
	print UInd(0,U)
	print UInd(1,U)
	print UInd(2,U)
	print
	print '---E---'
	print E(0,U)
	print E(1,U)
	print E(2,U)
	print
	print '---W---'
	print W(0,U,A)
	print W(1,U,A)
	print W(2,U,A)
	print
	print '---WDens---'
	print WDens(U[0],U,A)
	print WDens(U[1],U,A)
	print WDens(U[2],U,A)
	print WDens(U[3],U,A)
	print WDens(U[4],U,A)
	print WDens(U[5],U,A)
	
	
def main():

	U,A = getTestData()	
	#U,A = getLymphographyData()	
	#paperTest(U,A)
	#WDOD(U,A,0.4)
	WDODOtimizado(U,A,0.4)

	
if __name__ == '__main__':
    main()
