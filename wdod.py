from __future__ import division
import time
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
	return outliers
	
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
	
	return outliers
	
def getClassList(matrix, classPosition):
	
	classList = []
	
	for x in matrix:
		classList.append(x[classPosition])
		
	return classList
	
def getClassOfOutliers (outliers, data):

	classList = []
	rows, columns = data.shape
	
	for outlier in outliers:
		for element in data:
			elementWithoutClass = element
			elementWithoutClass = np.delete(elementWithoutClass,[columns-1])

			if np.array_equal(outlier,elementWithoutClass):
				classList.append(element[-1])
				break
			
	return classList

def getClassOfOutliersReversed (outliers, data):

	classList = []
	rows, columns = data.shape
	
	for outlier in outliers:
		for element in data:
			elementWithoutClass = element
			elementWithoutClass = np.delete(elementWithoutClass,[0])

			if np.array_equal(outlier,elementWithoutClass):
				classList.append(element[-1])
				break
			
	return classList
	
def getBreastCancerData():
	DATA = np.loadtxt('data/breast-cancer-wisconsin2.data',delimiter=',')
	DATA = DATA[:, 1:] # removes the first columns
	
	U = DATA[:, :-1]
	rows,columns = U.shape
	A = range(0,columns)
	
	return DATA,U,A
	
#NO LYMPH, AS RARE CLASSES SAO 1 e 4
def getLymphographyData():
	#data with class
	DATA = np.loadtxt('data/lymphography.all',delimiter=',')

	#U Without class
	U = DATA[:, :-1]
	rows,columns = U.shape
	A = range(0,columns)
	
	return DATA,U,A

def getTestData():
	U = np.array([["A","E","M"],["A","D","N"],["B","G","M"],["C","D","N"],["C","G","M"],["C","F","N"]])
	rows,columns = U.shape
	A = range(0,columns)
	
	return U,A
	
def getDataWithMultipliedRowsAndColumns(U, horizontalFactor, verticalFactor):

	copyU = U
	
	for x in range(1,horizontalFactor):
		copyU = np.vstack([copyU,U])
		
	for x in range(1,verticalFactor):
		copyU = np.hstack([copyU,U])
	return copyU
	
def rowsTest():
	DATA,U,A = getLymphographyData()
	print
	for x in range (1,10):
		
		thousandRows = getDataWithMultipliedRowsAndColumns(U,7*x,0)
		rows, columns = thousandRows.shape
		print rows, "rows"
		start_time = time.clock()
		outliers = WDODOtimizado(thousandRows,A,0.4)
		print time.clock() - start_time, "seconds"

def columnsTest():
	DATA,U,A = getLymphographyData()
	print
	for x in range (1,5):
		
		thousandColumns = getDataWithMultipliedRowsAndColumns(U,0,x)
		rows, columns = thousandColumns.shape
		print columns, "columns"
		start_time = time.clock()
		outliers = WDODOtimizado(thousandColumns,A,0.4)
		print time.clock() - start_time, "seconds"		
		
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

	#U,A = getTestData()
	#paperTest(U,A)
	#WDOD(U,A,0.4)
	
	
	DATA,U,A = getLymphographyData()
	outliers = WDODOtimizado(U,A,0.4) # - 4/4 unidades no lymph
	
	#outliers = WDODOtimizado(U,A,0.50)# - 6/8 unidades no lymph
	#outliers = WDODOtimizado(U,A,0.53)# - 12 unidades no lymph
	#outliers = WDODOtimizado(U,A,0.541)# - 15 unidades no lymph
	print 'Outliers candidates'
	print outliers
	print 'Classe dos outliers'
	print(getClassOfOutliers(outliers, DATA))
	print
	
	rowsTest()
	columnsTest()
	
	#DATA,U,A = getBreastCancerData()
	#outliers = WDODOtimizado(U,A,0.041) # - 5/5
	#outliers = WDODOtimizado(U,A,0.049) # - 15/16
	#print 'Classe dos outliers'
	#print(getClassOfOutliers(outliers, DATA))
	#print

	
if __name__ == '__main__':
    main()
