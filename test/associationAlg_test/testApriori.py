from programmingalpha.Utility.CorrelationAnalysis import Apriori,stepAprioriSearch

def loadData():
    data=[[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
    data=frozenset(map(frozenset,data))

    return data

dataSet = loadData()

apriori=Apriori()
apriori.minConfidence=0.8
apriori.minSupport=0.5
apriori.maxK=2
a,b = apriori.mineSupportSet(dataSet)

print(a)
print(b)

itemSeed=[{1},{3},{4}]
itemSeed=list(map(frozenset,itemSeed))
print("=="*20)
results=stepAprioriSearch(apriori,dataSet,itemSeed)
if results:
    print(len(results.keys()),results.keys())
    print(results)

    print()
    for item in results.keys():
        print("*"*20,item,itemSeed,item.intersection(itemSeed))

        if Apriori.containsAny(item,itemSeed):
            print(item,"<=",results[item])
