import numpy as np
import multiprocessing
import argparse
import programmingalpha
from functools import partial
import tqdm
import logging
import collections
import json
import time

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def _maxClip(data,maxSize):
    if len(data)>1.2*maxSize:
        np.random.shuffle(data)
        data=data[:maxSize]
    return data

def relationPCore(path_data,nodes,visited:set,use_intrisic=True):
    links=[]
    source=path_data["id"]
    targets_path=path_data["distances"]


    source=int(source)

    #print(source,visited,targets_path)
    #print(targets_path)
    #time.sleep(1)

    for target in targets_path:

        if frozenset({source,int(target)}) in visited:
            continue

        if targets_path[target]==0:
            target=int(target)
            links.append({"label":"duplicate","pair":(source,target)})
        elif targets_path[target]==1:
            target=int(target)
            links.append({"label":"direct","pair":(source,target)})
        elif targets_path[target]==2:
            target=int(target)
            links.append({"label":"transitive","pair":(source,target)})
        elif use_intrisic and targets_path[target]>2:
            target=int(target)
            links.append({"label":"unrelated","pair":(source,target)})
        else:
            raise ValueError("unexpected value {} for link relation".format(targets_path[target]))

        visited.add(frozenset({source,target}))

    if use_intrisic==False:
        link_num=len(links)

        add_ons=max(link_num//3,1)
        targets=set(map(lambda x:int(x),targets_path.keys()))
        targets.add(source)

        count=0
        for node in nodes:
            if count>add_ons:
                break
            if node in targets:
                continue
            links.append({"label":"unrelated","pair":(source,node)})
            count+=1

        #print("one is like =>",links)


    return links

def pairRelation(distanceData:list,nodes):
    labeled_link_file=programmingalpha.DataPath+"linkData/"+dbName.lower()+'-labelPair.json'

    counter=dict()
    linkData=[]
    print("final distance data")
    for d in distanceData[-10:]:
        print(d)

    #workers=multiprocessing.Pool(args.workers)

    with open(labeled_link_file,"w") as f:
        visited=set()
        for path in tqdm.tqdm(distanceData,desc="computing in batch"):
            links=relationPCore(path,nodes,visited)
            linkData.extend(links)

            if len(linkData)>args.batch_size:
                np.random.shuffle(nodes)
                tmp_labels=list(map(lambda ll:ll["label"],linkData))
                tmp_dict=dict(collections.Counter(tmp_labels))
                counter.update(tmp_dict)

                linkData_str=map(lambda labeled:json.dumps(labeled)+"\n",linkData)
                f.writelines(linkData_str)
                linkData.clear()



        if len(linkData)>0:
            tmp_labels=list(map(lambda ll:ll["label"],linkData))
            tmp_dict=dict(collections.Counter(tmp_labels))
            counter.update(tmp_dict)

            linkData_str=map(lambda labeled:json.dumps(labeled)+"\n",linkData)
            f.writelines(linkData_str)
            linkData.clear()


    #workers.close()
    #workers.join()

    logger.info("labels:{}".format(counter))

def main():

    maxPathLength=args.maxLength
    logger.info("loading distance data")
    distance_file=programmingalpha.DataPath+"linkData/"+dbName.lower()+'-%dgraph.json'%maxPathLength
    distance_data=[]
    nodes=[]
    with open(distance_file,"r") as f:
        for line in f:
            path=json.loads(line)
            distance_data.append(path)
            nodes.append(path["id"])

    pairRelation(distance_data,nodes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000000)
    parser.add_argument('--db', type=str, default="stackoverflow")
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--maxLength', type=int, default=2)

    args = parser.parse_args()

    dbName=args.db

    main()
