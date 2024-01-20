import collections

class UnifoldSet(object):
    def __init__(self,data_list):
        self.pre={}
        self.rank={}
        for node in data_list:
            self.pre[node]=node
            self.rank[node]=0

    def find(self,node):
        identifier=node
        while self.pre[identifier]!=identifier:
            identifier=self.pre[identifier]

        return identifier

    def join(self,node1,node2):
        identifier1=self.find(node1)
        identifier2=self.find(node2)
        if identifier1==identifier2:
            return

        if self.rank[identifier1]<self.rank[identifier2]:
            self.pre[identifier1]=identifier2
        elif self.rank[identifier2]<self.rank[identifier1]:
            self.pre[identifier2]=identifier1
        else:
            self.pre[identifier1]=identifier2
            self.rank[identifier2]+=1

    def getNumFolds(self):
        return collections.Counter(self.pre.values())

    def getFolds(self):
        folds={}
        for node, id in self.pre.items():
            if id not in folds:
                folds[id]=set()
                folds[id].add(node)
            else:
                folds[id].add(node)

        return folds
