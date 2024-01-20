import collections
import argparse
from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
import tqdm
import json
import multiprocessing
from functools import partial
import logging
import numpy as np

#from programmingalpha.Utility.DataStructure import UnifoldSet

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



def buildGraph(dbName):
    db=MongoStackExchange(host='10.1.1.9',port=50000)
    db.useDB(dbName)

    links=db.stackdb.get_collection("postlinks")

    allLinks=list(links.find().batch_size(args.batch_size))


    myG={}
    for link in tqdm.tqdm(allLinks,desc="building graph from links"):
        id_a,id_b=link["PostId"],link["RelatedPostId"]
        r=link["LinkTypeId"]
        if r==3:
            w=0
        elif r==1:
            w=1
        else:
            raise ValueError("unexpected value {} for link type".format(r))


        if id_a in myG:
            myG[id_a][id_b]=w
        else:
            myG[id_a]={id_b:w}

        if id_b in myG:
            myG[id_b][id_a]=w
        else:
            myG[id_b]={id_a:w}

    logger.info("finished finding {} sublinks".format(len(allLinks)))

    return myG

edges_counter=lambda records: sum(map(lambda x:len(x["distances"]),records))
def computeEdge(graphDict:dict):
    count=0
    for k in graphDict:
        count+=len(graphDict[k])
    return count
def showGraph(graphDict:dict, num=10):
    count=0
    for key in graphDict.keys():
        if count>num:
            break
        print(len(graphDict[key]),graphDict[key])
        count+=1


def init(gDict):
    global linkMap
    linkMap=gDict.copy()
    logger.info("init sub process: {}".format(multiprocessing.current_process()))

def computPathLenBFS(src,maxDistance,restrict=20):
    seq=collections.OrderedDict()
    visited=set()
    distances={}

    for node in linkMap[src].keys():
        if node ==src:
            continue
        seq[node]=linkMap[src][node]

    while len(seq)>0 and len(distances)<restrict:

        node,length=seq.popitem(False)
        visited.add(node)
        if length>maxDistance:
            distances[node]=length

        if node not in linkMap[src].keys():
            linkMap[src][node]=length

        for nb in linkMap[node].keys():
            if nb in seq or nb in visited:
                continue

            dist=length+linkMap[node][nb]

            seq[nb]=dist

    #if len(distances)>0:
    #   print(len(distances),distances)
    return {"id":src,"distances":distances}

def computeViaBFS(graphDict:dict,need_num):
    extra_distances_data=[]

    nodes=list(graphDict.keys())
    np.random.shuffle(nodes)
    _compute=partial(computPathLenBFS,maxDistance=args.maxLength)
    init(graphDict)
    count=0

    for src in tqdm.tqdm(nodes,desc="computing {} more edges with len> maxlength({}) using bfs".format(need_num,args.maxLength)):
        #print("current edges",count)
        if count>need_num:
            #print(count)
            break
        record =_compute(src)
        extra_distances_data.append(record)
        count+=len(record["distances"])

    return extra_distances_data

def computeViaBFSParallel(graphDict:dict,need_num):
    extra_distances_data=[]

    nodes=list(graphDict.keys())
    np.random.shuffle(nodes)
    _compute=partial(computPathLenBFS,maxDistance=args.maxLength)
    count=0
    batches=[nodes[i:i+args.batch_size] for i in range(0,len(nodes),args.batch_size)]
    workers=multiprocessing.Pool(args.workers,initializer=init,initargs=(graphDict,))
    verbose=1
    for srcs in tqdm.tqdm(batches,desc="computing {} more edges with len> maxlength({}) using bfs".format(need_num,args.maxLength)):
        #print("current edges",count)
        if count>need_num:
            #print(count)
            break
        if count/need_num>0.05*verbose:
            logger.info("{}%".format(0.05*verbose*100))
            verbose+=1

        for record in workers.map(_compute,srcs):
            extra_distances_data.append(record)
            count+=len(record["distances"])

    workers.close()
    workers.join()
    return extra_distances_data


def computePathLenDP(src,maxDistance):
    nodes=set(linkMap[src].keys())

    for node in nodes:

        tgts=linkMap[node].keys()

        for tgt in tgts:
            if tgt ==src or tgt in nodes:
                continue

            dist=linkMap[src][node]+linkMap[node][tgt]

            if dist<=maxDistance:
                linkMap[src][tgt]=linkMap[tgt][src]=dist

    distances={src:linkMap[src]}

    return distances

def computeAllPairDistance(graphDict:dict):
    nodes=list(graphDict.keys())


    iterNum=3

    edges_seq=[computeEdge(graphDict)]

    for _ in range(iterNum):

        _compute=partial(computePathLenDP,maxDistance=args.maxLength)
        init(graphDict)

        for src in tqdm.tqdm(nodes,desc="computing from nodes"):

            distances = _compute(src)
            graphDict.update(distances)

        edges_seq.append(computeEdge(graphDict))
        print("after one epoch",edges_seq)

        np.random.shuffle(nodes)

        if edges_seq[-1]==edges_seq[-2]:
            break

    return graphDict

def computeAllPairDistanceParallel(graphDict:dict):
    nodes=list(graphDict.keys())

    batches=[nodes[i:args.batch_size] for i in range(0,len(nodes),args.batch_size)]

    workers=multiprocessing.Pool(args.workers,initargs=(graphDict,),initializer=init)

    iterNum=300

    edges_seq=[computeEdge(graphDict)]

    for _ in range(iterNum):
        np.random.shuffle(nodes)

        _compute=partial(computePathLenDP,maxDistance=args.maxLength)

        for batch_srcs in tqdm.tqdm(batches,desc="computing from nodes in parallel"):
            #print("one batch")
            #distances=computePathLenDP(src,graphDict,args.maxLength)
            for distances in workers.map(_compute,batch_srcs):
                graphDict.update(distances)

        edges_seq.append(computeEdge(graphDict))
        logger.info("after {} epoch=>{}".format(len(edges_seq)-1,edges_seq[-10:]))
        #showGraph(graphDict)


        if edges_seq[-1]==edges_seq[-2]:
            break

    workers.close()
    workers.join()

    return graphDict




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--db', type=str, default="stackoverflow")
    parser.add_argument('--workers', type=int, default=28)
    parser.add_argument('--maxLength', type=int, default=2)

    args = parser.parse_args()

    dbName=args.db

    graphDict=buildGraph(dbName)

    #'''
    edges_num=computeEdge(graphDict)
    print("init edges num",edges_num)
    print("before")
    showGraph(graphDict)

    graphDict=computeAllPairDistanceParallel(graphDict)

    print("after")
    showGraph(graphDict)
    #'''

    distances_data=[]
    for k in graphDict.keys():
        record={"id":k,"distances":graphDict[k]}
        distances_data.append(record)

    edges_num=edges_counter(distances_data)
    need_num=edges_num//3

    print("found {} edges records({})".format(edges_num,len(distances_data)))

    extra_data=computeViaBFSParallel(graphDict,need_num)
    print("extra edges found, records({})",edges_counter(extra_data),len(extra_data))

    distances_data.extend(extra_data)

    edges_num=edges_counter(distances_data)
    print("final edges",edges_num,"records",len(distances_data))
    #exit(10)

    distance_file=programmingalpha.DataPath+"linkData/"+dbName.lower()+'-%dgraph.json'%args.maxLength
    with open(distance_file,"w") as f:
        distances_data_str=map(lambda record:json.dumps(record)+"\n",distances_data)
        f.writelines(distances_data_str)

