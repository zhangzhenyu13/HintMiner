from programmingalpha.Utility.CorrelationAnalysis import *

def loadSimpDat():
    simDat = [['r','z','h','j','p'],
              ['z','y','x','w','v','u','t','s'],
              ['z'],
              ['r','x','n','o','s'],
              ['y','r','x','z','q','t','p'],
              ['y','z','x','e','q','s','t','m']]
    return simDat
# 构造成 element : count 的形式
def createInitSet(dataSet):
    retDict={}
    for trans in dataSet:
        key = frozenset(trans)
        if key in retDict.keys():
            retDict[frozenset(trans)] += 1
        else:
            retDict[frozenset(trans)] = 1
    return retDict

simDat = loadSimpDat()
initSet = createInitSet(simDat)
fpgrowth=FPTree()
myFPtree, myHeaderTab = fpgrowth.createFPTree(initSet, 3) # 最小支持度3

def showInfo():
    myFPtree.showSubTree(0)
    for item,header in myHeaderTab.items():
        print(item,header[0],"+"*30)
        header=header[1]
        while header is not None:
            print(header.name,header.frequent)
            header=header.nextLink


def testPaths():
    print (fpgrowth.findPrefixBases('z',))
    print (fpgrowth.findPrefixBases('r'))
    print (fpgrowth.findPrefixBases('x'))
    print("-"*30)

def testfreqmine():
    fqitems=mineFPTree(3,fpgrowth)

    for items,c in fqitems.items():
        print(c,items)


#showInfo()

#testPaths()

testfreqmine()

