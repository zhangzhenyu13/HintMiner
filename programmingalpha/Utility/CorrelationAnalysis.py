import numpy as np
import time
import collections

class CorrelationMiner(object):

    extraCases=set()

    def __init__(self):
        self.minSupport=1000
        self.minConfidence=0.8

    @staticmethod
    def containsAny(mySet,conatainer):
        "judge if any of the container element is appeared as a subset of mySet or an element of mySet"

        for data in conatainer:
            if type(data) in [list,set,frozenset] and len(set(data).intersection(mySet))>0:
                return True
            elif data in mySet:
                return True

        return False


class Apriori(CorrelationMiner):
    def __init__(self):
        CorrelationMiner.__init__(self)
        self.maxK=np.inf


    def initSingleItem(self,dataSet):
        c1 = []
        for transaction in dataSet:
            for item in transaction:
                if  [item] not in c1:
                    c1.append([item])

        return list(map(frozenset,c1))

    def scanDataset(self,D,Ck):
        #returns next K set and support count data

        supportCount = {}
        retList = []

        if len(Ck)<1:
            return retList, supportCount

        def countForCan(can):
            count=0
            for tid in D:
                if can.issubset(tid):
                    count+=1
            supportCount[can]=count

        vectorizerSearch=np.vectorize(countForCan)
        #print(type(Ck),len(Ck))
        vectorizerSearch(Ck)


        numItems = float(len(D))

        for key in supportCount:
            support = supportCount[key]
            if support >= self.minSupport or self.containsAny(key,self.extraCases):
                retList.append(key)

        return retList,supportCount

    def aprioriGen(self,Lk,k):
        #generate candidates

        retList = []
        lenLk = len(Lk)

        #print("current K",k)

        for i in range(lenLk):
            for j in range(i+1,lenLk):
                L1 = Lk[i]
                L2 = Lk[j]
                L=L1|L2
                #print("L1/L2",L1,L2)
                if len(L)>k:
                    #print("to skip", len(L),L)
                    continue

                symL=L1.symmetric_difference(L2)

                if k>2 and symL not in Lk:
                    #print("symetric to skip",L1.symmetric_difference(L2))
                    continue
                retList.append(L)

        return retList

    def mineSupportSet(self,dataSet):
        C1 = self.initSingleItem(dataSet)

        L1,supportData = self.scanDataset(dataSet,C1)

        L = [L1]

        k = 2

        while (len(L[k-2]) > 0) and k<=self.maxK :
            Ck = self.aprioriGen(L[k-2],k)
            if len(Ck)<1:
                break
            Lk,supK = self.scanDataset(dataSet,Ck)
            if len(Lk)>0:
                supportData.update(supK)
                L.append(Lk)
            else:
                break
            k += 1
        return L,supportData


def stepAprioriSearch(apriori:Apriori,dataSet,itemSeed):

        t0=time.time()

        C1=apriori.initSingleItem(dataSet)

        C1=set(itemSeed).union(C1)
        C1=list(C1)

        print("searching with %d candidates, %d seeds"%(len(C1),len(itemSeed)))

        L1, supportData=apriori.scanDataset(dataSet,C1)

        print("generate data from related %d L1 (cost=%ds)"%(len(L1),time.time()-t0))

        C2=apriori.aprioriGen(L1,2)

        print("generated %d candidates for L2 (cost=%ds)"%(len(C2),time.time()-t0))

        if len(C2)<1:
            return None

        L2,sup2=apriori.scanDataset(dataSet,C2)

        print("get %d result records (cost=%ds)"%(len(L2),time.time()-t0))

        supportData.update(sup2)

        frequentItems=supportData

        return frequentItems

########Fp_growth Tree

class TreeNode(object):
    def __init__(self,nodeName,nodeCount,ancester):
        self.name=nodeName
        self.frequent=nodeCount
        self.ancester=ancester

        self.nextLink=None
        self.children={} #name,TreeNode

    def increaseNum(self,addCount):
        self.frequent+=addCount

    def showSubTree(self,depth=1):
        queue=collections.deque()
        queue.append(self)
        while len(queue)>0:
            node=queue.popleft()
            print("*"*depth+"%s,%d"%(node.name,node.frequent))

            for child in self.children.values():
                child.showSubTree(depth+1)

class FPTree(CorrelationMiner):

    def __init__(self):
        CorrelationMiner.__init__(self)
        self.headerTable={}
        self.tree=TreeNode("ROOT-NULL",0,None)

    def updateHeaderTable(self,header:TreeNode,targetNode:TreeNode):
        while header.nextLink is not None:
            header=header.nextLink
        header.nextLink=targetNode

    def updateFPTree(self,items,tree:TreeNode,headerTable,count):
        'all the items in in the itenms appeared for count times'
        ptr=0
        root=tree
        while len(items[ptr:])>0:
            item=items[ptr]

            if item in root.children.keys():
                root.children[item].increaseNum(count)
            else:
                root.children[item]=TreeNode(item,count,root)
                if headerTable[item][1] is None:
                    #add first header link
                    headerTable[item][1]=root.children[item]
                else:
                    self.updateHeaderTable(headerTable[item][1],root.children[item])

            root=root.children[item]
            ptr+=1

    def createFPTree(self,dataSet,minSup):
        #print("building fp tree using")
        #for d,c in dataSet.items():
        #    print(c,d)

        #data loader

        for trans,count in dataSet.items():
            for item in trans:
                if item not in self.headerTable.keys():
                    self.headerTable[item]=[count,None]
                else:
                    self.headerTable[item][0]+=count

        item_to_Remove=[]
        for item,count in self.headerTable.items():
            if count[0]<minSup and item not in self.extraCases:
                #print("do not remove as it is extra-keeped",item,"{}:({})".format(len(self.extraCases),self.extraCases))
                item_to_Remove.append(item)

        #if len(item_to_Remove)==len(self.headerTable.keys()):
            #print("all items are not frequent")

        for item in item_to_Remove:
            if item in self.extraCases:
                print("remove", item)
            del self.headerTable[item]

        if len(self.headerTable)==0:
            return None,dict()

        frequentItems=set(self.headerTable.keys())

        #print("scanned the dataset and find %d frequentf items, now begin to to build fp-tree"%len(frequentItems))

        for items, count in dataSet.items():

            filtered_items=[]

            for item in items:
                if item in frequentItems:

                    filtered_items.append((item,self.headerTable[item][0]))

            if len(filtered_items)>0:
                #print("before",filtered_items)
                filtered_items.sort(key=lambda x:(x[1],x[0]),reverse=True)
                items=[v[0] for v in filtered_items]
                #print("after",items)
                #print("adding to fp tree",items)

                self.updateFPTree(items,self.tree,self.headerTable,count)

                #print("current tree"+"="*20)
                #root.showSubTree(0)

        #print("finished builing fp-tree")

        return self.tree,self.headerTable

    def findPrefixBases(self,basePat):
        #print("search base in header table",basePat,self.headerTable.keys())

        baseNode=self.headerTable[basePat][1]
        condPats={} # base path: support of the base path for the basePat

        def findLeafPrefix(leafNode:TreeNode):
            prefixPath=[]

            while leafNode is not None:
                prefixPath.append(leafNode.name)
                leafNode=leafNode.ancester

            prefixPath.pop()
            return prefixPath

        while baseNode is not None:
            prefixPath=findLeafPrefix(baseNode)
            if len(prefixPath)>1:
                condPats[frozenset(prefixPath[1:])]=baseNode.frequent # as the base node has the smallest support

            baseNode=baseNode.nextLink

        return condPats


def mineFPTree(minSup,fpTree:FPTree,maxKlen=np.inf):

    prefix=set()
    frequentItemsList={}

    mineTasks=collections.deque()
    mineTasks.append([fpTree,prefix])

    while len(mineTasks)>0:
        fpTree,prefix=mineTasks.popleft()

        headersT = fpTree.headerTable
        basePats=[v[0] for v in sorted(headersT.items(),key=lambda p:p[0],reverse=True)]

        #print("sorted %d base patterns"%len(basePats),basePats)

        for basePat in basePats:
            #print("from ",basePat)
            newFreqSet=prefix.copy()
            newFreqSet.add(basePat)
            frequentItemsList[frozenset(newFreqSet)]=headersT[basePat][0]
            condBases=fpTree.findPrefixBases(basePat)

            if len(condBases)>0 and len(newFreqSet)<maxKlen:
                nextfpTree=FPTree()
                nextfpTree.createFPTree(condBases,minSup)
                if len(nextfpTree.headerTable)>0:
                    mineTasks.append([nextfpTree,newFreqSet.copy()])
                    #print("added headers",nextfpTree.headerTable.keys(),"current prefix",newFreqSet)

            #print()
    return frequentItemsList
