from programmingalpha.DataSet.DBLoader import MongoStackExchange
import argparse
import numpy as np
import tqdm
import programmingalpha
import os
import json
import logging
from collections import Counter
import random

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)


def recoverSent(texts,sep=" ",tokenizer=None):
    text=" ".join(texts)
    if tokenizer is None:
        text=" ".join(text.split())
    else:
        text=sep.join(tokenizer.tokenize(text))


    return text


def inferenceGen():

    collection=docDB.stackdb["inference"]
    duplicates=list(collection.find({"label":"duplicate"}).batch_size(args.batch_size))

    size=len(duplicates)
    if args.maxSize>0 and args.maxSize<=size*4:
        size=args.maxSize//4

    query1=[
          {"$match": {"label":"duplicate"}},
          {"$sample": {"size": size}}
        ]

    query2=[
          {"$match": {"label":"direct"}},
          {"$sample": {"size": size}}
        ]

    query3=[
          {"$match": {"label":"transitive"}},
          {"$sample": {"size": size}}
        ]

    query4=[
          {"$match": {"label":"unrelated"}},
          {"$sample": {"size": size}}
        ]
    queries=[query1,query2,query3,query4]

    dataSet=[]

    labels=[]
    for query in queries:
        data=[]
        data_samples=list(collection.aggregate(pipeline=query,allowDiskUse=True))
        for record in tqdm.tqdm(data_samples,desc="{}".format(query)):
            del record["_id"]
            record["q1"]=recoverSent(record["q1"])
            record["q2"]=recoverSent(record["q2"])
            if len(record["q1"].split())<20 or len(record["q2"].split())<20:
                continue
            labels.append(record["label"])
            data.append(json.dumps(record)+"\n")


        dataSet.extend(data)
        data.clear()

    logger.info("laebls:{}".format(Counter(labels)))

    np.random.shuffle(dataSet)

    inference_sample_file=os.path.join(programmingalpha.DataPath,"inference/data.json")

    logger.info("saving data to "+inference_sample_file)
    with open(inference_sample_file,"w") as f:
        f.writelines(dataSet)



def seq2seqGen():
    collection=docDB.stackdb["seq2seq"]
    size=collection.count()
    if args.maxSize>0 and args.maxSize<size:
        size=args.maxSize

    data_samples=list(collection.find().limit(size).batch_size(args.batch_size))
    #print(data_samples[:2])

    dataSet=[]
    for record in tqdm.tqdm(data_samples,desc="retriving seq2seq samples(size)".format(size)):
        del record["_id"]
        if len(recoverSent(record["answer"]).split())<10 or len(recoverSent(record["context"]).split())<20 or \
                len( recoverSent(record["question"]).split())<10:
            continue
        dataSet.append(record)


    def _constructSrc(record):
        question=recoverSent(record["question"])
        context=recoverSent(record["context"],sep="[SEP]").split()[:args.contextLen+len(record["context"])-1]
        context=" ".join(context).split("[SEP]")

        pos=random.randint(0,len(context))
        answer=recoverSent(record["answer"]).split()[:args.answerLen]
        answer=" ".join(answer)
        context=recoverSent(context[:pos])+answer+recoverSent(context[pos:])

        seq_src=[]
        question_tokens=question.split()[:args.questionLen]
        context_tokens=context.split()

        seq_src+=question_tokens
        seq_src+=["[SEP]"]
        seq_src+=context_tokens
        seq_src+=["[SEP]"]

        return " ".join(seq_src)+"\n"

    def _constructDst(record):
        answer=recoverSent(record["answer"])
        answer_tokens=answer.split()[:args.answerLen]
        seq_tgt=[]
        seq_tgt+=answer_tokens


        return " ".join(seq_tgt)+"\n"

    logger.info("data size={}".format(len(dataSet)))
    dataSrc=map(_constructSrc,dataSet)
    dataDst=map(_constructDst,dataSet)


    seq2seq_sample_file_src=os.path.join(programmingalpha.DataPath,"seq2seq/data-src")
    seq2seq_sample_file_dst=os.path.join(programmingalpha.DataPath,"seq2seq/data-dst")


    logger.info("saving data to "+seq2seq_sample_file_src)
    with open(seq2seq_sample_file_src,"w") as f:
        f.writelines(dataSrc)

    logger.info("saving data to "+seq2seq_sample_file_dst)
    with open(seq2seq_sample_file_dst,"w") as f:
        f.writelines(dataDst)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--db', type=str, default="corpus")
    parser.add_argument('--maxSize', type=int, default=-1)
    parser.add_argument('--task', type=str, default="seq2seq")
    parser.add_argument('--contextLen', type=int, default=1000)
    parser.add_argument('--questionLen', type=int, default=100)
    parser.add_argument('--answerLen', type=int, default=250)

    args = parser.parse_args()

    docDB=MongoStackExchange(host='10.1.1.9',port=50000)
    docDB.useDB(args.db)

    if args.task=="inference":
        logger.info("task is "+args.task)
        inferenceGen()
    if args.task=="seq2seq":
        logger.info("task is "+args.task)
        seq2seqGen()
