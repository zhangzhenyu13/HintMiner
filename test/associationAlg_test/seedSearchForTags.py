import argparse
from programmingalpha.DataSet.DBLoader import MongoStackExchange
from programmingalpha.Utility.CorrelationAnalysis import Apriori,stepAprioriSearch,mineFPTree,FPTree
import numpy as np
import programmingalpha
import json,pickle

#for pair tags indexer 0/1
otherTagIndex=lambda tag: 1 if tag[0] in m_tag else 0 #find the otehr tag not in m_tag
seedTagIndex=lambda tag :0 if tag[0] in m_tag else 1 #find tag that is in m_tag


class TagCounter(object):
    def loadTransactionData(self):
        questions=self.mongoDB.questions.find().batch_size(args.batch_size)

        for q in questions:
            self.transactions.append(q["Tags"])
            if len(self.transactions)%(int(100*args.batch_size))==0:
                print("scanned %d questions"%(len(self.transactions)))

            if tuneMaxClipNum and len(self.transactions)>=tuneMaxClipNum:
                break

        print("scanned %d questions"%(len(self.transactions)))

        self.transactions=list(map(frozenset,self.transactions))

    def __init__(self, mongodb:MongoStackExchange, dbName:str):
        self.mongoDB=mongodb
        self.mongoDB.useDB(dbName)
        self.transactions=[]

        self.ItemSeeds=set()

    def getRelatedTagsViaApriori(self):

        Apriori.extraCases=set(m_tag)
        apriori=Apriori()

        apriori.maxK=2
        apriori.minSupport=1000
        apriori.minConfidence=0.7

        frequentItems=stepAprioriSearch(apriori,self.transactions,self.ItemSeeds)

        return frequentItems

    def __FPGInputData(self,dataSet):
        retDict={}
        for trans in dataSet:
            key = frozenset(trans)
            if key in retDict.keys():
                retDict[frozenset(trans)] += 1
            else:
                retDict[frozenset(trans)] = 1
        return retDict

    def getRelatedTagsViaFPTG(self):

        FPTree.extraCases=set(self.ItemSeeds)
        fpTree=FPTree()
        fpTree.maxK=2
        fpTree.minSupport=0
        fpTree.minConfidence=0.7

        fpTree.createFPTree(self.__FPGInputData(self.transactions),minSuppport)
        frequentItems=mineFPTree(minSuppport,fpTree,maxKlen=2)

        return frequentItems

    def getConfidenceData(self,frequentItems):
        #print("frequentItems({})".format(len(frequentItems)),frequentItems)
        #pair frequentItems

        confData={}
        seedsDict={}

        for seed in self.ItemSeeds:
            seed=frozenset({seed})
            if seed not in frequentItems.keys():
                print("seed dict key missing")
                seedsDict[seed]=0
            else:
                seedsDict[seed]=frequentItems[seed]

        print("seed dict(%d)"%len(seedsDict),seedsDict)

        def keepDict(items,sup):
            #print("entry",entry)
            #print("val-input",items,sup)
            #items,sup=entry
            if len(items)>1 and len(items.intersection(m_tag))==1:
                tagData=list(items)
                seedIndex=seedTagIndex(tagData)
                seed={tagData[seedIndex]}
                seed=frozenset(seed)
                otherTag=frozenset({tagData[1-seedIndex]})
                try:
                    confData[items]=[sup/seedsDict[seed] if seedsDict[seed]>0 else 0,
                                     sup/frequentItems[otherTag]]
                except:
                    print("***%%%tuning=>",tagData[0],tagData[1],seedTagIndex(tagData),seed,tagData)
                    pass

        vectorizer=np.vectorize(keepDict)

        itemsL=[]
        supL=[]
        for k,v in frequentItems.items():
            itemsL.append(k)
            supL.append(v)

        vectorizer(itemsL,supL)


        return confData

    def getTagCounter(self,relatedTags):
        tag_collection=self.mongoDB.tags
        counter={}

        for tagName in relatedTags:
            tagName=tagName[1:][:-1]
            tag=tag_collection.find_one({"TagName":tagName})

            if tag:
                count=tag["Count"]
            else:
                #print("not found",tag)
                #exit(10)
                continue

            #print(tagName,count)

            counter["<{}>".format(tagName)]=count

        return counter

def getFrequentItems(load=True):
    if not load:
        tagCounter.loadTransactionData()
        frequentItems=tagCounter.getRelatedTagsViaFPTG()
        with open(programmingalpha.DataPath+"frequentItems.pickle","wb") as f:
            pickle.dump(frequentItems,f)
    else:
        with open(programmingalpha.DataPath+"frequentItems.pickle","rb") as f:
            frequentItems=pickle.load(f)

    return frequentItems


if __name__ == '__main__':

    #settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongodb', type=str, help='mongodb host, e.g. mongodb://10.1.1.9:27017/',default='mongodb://10.1.1.9:27017/')
    parser.add_argument('--batch_size', type=str, default=10000)

    args = parser.parse_args()

    #load m_tags
    m_tag=programmingalpha.loadConfig(programmingalpha.ConfigPath+"TagSeeds.json")["Tags"]

    if not m_tag:
        m_tag=["<keras>","<tensorflow>","<caffe>","<pytorch>","<artificial-intelligence>","<nlp>","<computer-vision>",
               "<deep-learning>","<neural-network>","<machine-learning>","<reinforcement-learning>","<scikit-learn>"]
    print("search init with %d tag seeds"%len(m_tag))

    minSuppport=1000

    tuneMaxClipNum=None

    mongodb=MongoStackExchange(args.mongodb)
    dbname="stackoverflow"
    tagCounter=TagCounter(mongodb,dbname)
    tagCounter.ItemSeeds.update(m_tag)

    #mine fp tree
    frequentItems=getFrequentItems()

    print("get %d frequent patterns"%len(frequentItems),"below are the frequent patterns of seeds")

    '''
    seedCounter=tagCounter.getTagCounter(m_tag)
    for i in range(len(m_tag)):
        for j in range(1,len(m_tag)):
            tag1,tag2=m_tag[i],m_tag[j]
            tagP=frozenset({tag1,tag2})
            tag1=frozenset({tag1})
            tag2=frozenset({tag2})
            sup1=frequentItems[tag1] if tag1 in frequentItems else seedCounter[m_tag[i]]
            sup2=frequentItems[tag2] if tag2 in frequentItems else seedCounter[m_tag[j]]
            comsup=frequentItems[tagP] if tagP in frequentItems else 0
            conf1=comsup/sup1
            conf2=comsup/sup2
            print(tag1,tag2,tagP,sup1,conf2,comsup,conf1,conf2)
    '''

    print("+"*60)

    confData=tagCounter.getConfidenceData(frequentItems)
    print("find {} related tags' (using confidence computation)".format(len(confData)))


    print("confidence info")
    for k in confData:
        confidence=confData[k]
        if confidence[0]>0.2 or confidence[1]>0.2:
            print(k, confidence,frequentItems[k])
    print("-"*60)

