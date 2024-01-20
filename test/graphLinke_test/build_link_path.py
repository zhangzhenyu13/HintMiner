import networkx as nx
import argparse
from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
import tqdm
import json
import multiprocessing
from functools import partial
import logging
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

    G=nx.Graph()

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

        G.add_edge(id_a,id_b,weight=w)

        if id_a in myG:
            myG[id_a][id_b]=w
        else:
            myG[id_a]={id_b:w}

        if id_b in myG:
            myG[id_b][id_a]=w
        else:
            myG[id_b]={id_a:w}

    logger.info("finished finding {} sublinks".format(len(allLinks)))
    logger.info("graph size of edges({}) and nodes({})".format(len(list(G.edges)),len(list(G.nodes))))



    if len(G.nodes)<1e+4:
        return [G],G
    else:
        logger.info("cutting graph into small blocks")

    graphs=[]

    for cc in nx.connected_components(G):
        g=G.subgraph(cc)
        graphs.append(g)

    graphs.sort(key=lambda g:len(g.nodes),reverse=True)

    logger.info("num of subGs:{}".format(len(graphs)))
    subnodes=list(map(lambda g:len(g.nodes),graphs))[:10]
    logger.info("nodes of subG(top10):{}".format(subnodes))


    return graphs,G

def computePathCore(src,maxLength,G):
    pathLen=nx.single_source_dijkstra_path_length(G=G,source=src,cutoff=maxLength,weight="weight")

    if src in pathLen:
        del pathLen[src]
    return {"id":src,"distances":pathLen}

def computeBatch(srcs,maxLength,G):
    batch=[]
    for src in srcs:
        distance=computePathCore(src,maxLength,G)
        if len(distance["distances"])>0:
            batch.append(distance)
    return batch

def computeShortestPathParallel(G,maxLength=2):
    distanceData=[]
    worker_num=args.workers if args.workers>0 else multiprocessing.cpu_count()
    workers=multiprocessing.Pool(worker_num)
    _compute=partial(computeBatch,maxLength=maxLength,G=G)

    nodes=list(G.nodes)
    batch_size=args.batch_size
    batches=[nodes[i:i+batch_size] for i in range(0,len(nodes),batch_size)]

    step_batches=[batches[i:i+worker_num] for i in range(0,len(batches),worker_num)]

    logger.info("parallel configiuration as batch_size({}),step/worker_num({}), bateches({}), steps({})".format(
        batch_size,worker_num,len(batches),len(step_batches)
    ))



    for i in range(0,len(step_batches)):
        logger.info("computing step #{}".format(i+1))
        step_batch=step_batches[i]
        for batch in workers.map(_compute,step_batch):
            distanceData.extend(batch)

    workers.close()
    workers.join()

    return distanceData

def computeShortestPath(G,maxLength=2):
    distanceData=[]
    _compute=partial(computePathCore,maxLength=maxLength,G=G)

    for src in G.nodes:
        distance=_compute(src)
        if len(distance["distances"])>0:
            distanceData.append(distance)

    return distanceData


def main():

    logger.info("building graph link data for {}".format(dbName))

    maxPathLength=args.maxLength
    graphs,G=buildGraph(dbName)

    distance_data=[]
    for G in graphs:
        if len(G.nodes)>1000:
            logger.info("subgraph nodes: {}".format(len(G.nodes)))
        if len(G.nodes)>args.batch_size:
            distances=computeShortestPathParallel(G,maxPathLength)
        else:
            distances=computeShortestPath(G,maxPathLength)

        distance_data.extend(distances)

    if len(distance_data)>0:
        distance_file=programmingalpha.DataPath+"linkData/"+dbName.lower()+'-%dgraph.json'%maxPathLength
        with open(distance_file,"w") as f:
            distance_data_str=map(lambda distance:json.dumps(distance)+"\n",distance_data)
            f.writelines(distance_data_str)

    logger.info("shortest distance computing finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--db', type=str, default="AI")
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--maxLength', type=int, default=2)

    args = parser.parse_args()

    dbName=args.db

    main()
