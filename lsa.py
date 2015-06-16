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
import sys
import numpy as np

def lsa(D, k):
    counter = 0
    dRows, dColumns = D.shape
    D = np.c_[D, np.zeros(dRows)]
    #labelIndices = np.zeros(dRows)

    # Phase 1-initialization
    for index in xrange(dRows):
        counter += 1
        if counter <= k:
            putLabel(D[index], 'O')
        else:
            putLabel(D[index], 'N')
    # at this point, D[k+1:] is compound of non-outliers
    # and we need to create m  dicts, one for each attribute with their values and frequences
    newD = D[k:D.shape[0], :]
    attributesDict = {}
    for columns in xrange(newD.shape[1] - 2):
        attributesDict[columns] = createFrequencyDict(columns, newD)
        attributesDict[columns]['unique'] = np.unique(D[:,columns])

    # Phase 2- Iteration
    #

    globalEntropy = sys.maxint
    not_moved = True
    while not_moved:
        not_moved = False
        for di in D:
            if getLabel(di) == 'N':
                for dj in D:
                    if getLabel(dj) == 'O':
                        exchangeLabels(di, dj)
                        attributesDict = changeAttributesDict(attributesDict, di[:-2], dj[:-2])
                        currentEntropy = getEntropy(attributesDict)
#                        print 'before',  entropia1
#                        print 'after', entropia2
#                        print 'gloabl entropy', globalEntropy
                        #print 'outliers', D[D[:,-1] == 1]
                        if currentEntropy < globalEntropy :
                            globalEntropy = currentEntropy
                            not_moved = True
                        else:
                            exchangeLabels(di, dj)
                            attributesDict = changeAttributesDict(attributesDict, dj[:-2], di[:-2])

    return D


def getLymphographyData():
	#data with class
	DATA = np.loadtxt('data/lymphography.all',delimiter=',')

	#U Without class
	U = DATA[:, :-1]
	rows,columns = U.shape
	A = range(0,columns)
	
	return DATA,U,A

def getEntropy(attributesDict):
    entropyTotal = 0.0

    #print attributesDict 
    for key in attributesDict.keys():
        newDict = attributesDict.get(key)
        total_attributes = 0.0
        for second_key in newDict.keys():
            if second_key != 'unique':
                total_attributes += newDict.get(second_key)
        entropyLocal = 0
        for second_key in newDict.keys():
            if second_key != 'unique':
                num =newDict.get(second_key) 
                p = num/total_attributes
                
                if p <= 0 :
                    entropyLocal += 0
                else:
                    entropyLocal += p*np.log(p)
        entropyTotal += -entropyLocal
    return entropyTotal


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

def changeAttributesDict(attributeDict, di, dj):
            
    for index in xrange(di.shape[0]):
        newDict = attributeDict.get(index)
            #tirar di
#        apagou= True
#        if di[index] in newDict:
#            frequency = newDict[di[index]] - 1
#            newDict.update({di[index]:frequency})
#        else:
#            apagou = False
#
#        if apagou:
#            if dj[index] in newDict:
#                frequency = newDict[dj[index]] + 1
#                newDict.update({dj[index]: frequency})
#            else:
#                if dj[index] in newDict.get('unique'):
#                    newDict.update({dj[index]:1})
                
        apagou = True
        if di[index] in newDict:
            frequency = newDict[di[index]] - 1
            newDict.update({di[index]:frequency})
        else: 
            apagou = False
        # e adicionar dj
        if apagou:
            if dj[index] in newDict:
                frequency = newDict[dj[index]] + 1
                newDict.update({dj[index]: frequency})


    return attributeDict


def exchangeLabels(di, dj):
    temp = dj[-1]
    dj[-1] = di[-1]
    di[-1] = temp

def putLabel(d, label):

    if label == 'N':
        d[-1] =0
    elif label == 'O':
        d[-1] = 1

    return d


def getLabel(d):

    if d[-1] == 1:
        return 'O'
    elif d[-1] == 0:
        return 'N'


def getBreastCancerData():
        database1 = np.loadtxt('data/breast-cancer-wisconsin.data',delimiter=',')
        database1 = database1[:, 1:]
        (rows, columns) = database1.shape

        benign = np.array([]).reshape(0,columns)
        malign = np.array([]).reshape(0,columns)

        for d in database1:
            if d[-1] == 2:
                benign = np.vstack([benign, d])
            else:
                malign = np.vstack([malign, d])

        reducedMalign = malign [:39,:]
        benign = benign[:444,:]
        U = np.vstack([benign,reducedMalign])
        np.random.shuffle(U)
        return U

def main():
    #D = np.array([['a', 'e', 'm'], ['a', 'd', 'n'], ['b', 'g', 'm'], ['c', 'd', 'n'], ['c', 'g', 'm'], ['c', 'f', 'n']])
    #D = getBreastCancerData()
    D, U, A = getLymphographyData()
    outliers =  lsa(D,16)
    print outliers[outliers[:,-1] == 1]
    #print outliers

if __name__ == '__main__':
    main()
