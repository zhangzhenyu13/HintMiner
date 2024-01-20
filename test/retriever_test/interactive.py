#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf DrQA retriever module."""

import argparse
import code
import prettytable
import logging
from programmingalpha import retrievers
from programmingalpha.DataSet import DBLoader
import heapq
import programmingalpha

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
args = parser.parse_args()

logger.info('Initializing ranker...')


rankers={}
rankers['stackoverflow']=retrievers.get_class('tfidf')('stackoverflow')
rankers['AI']=retrievers.get_class('tfidf')('AI')
rankers['datascience']=retrievers.get_class('tfidf')('datascience')
rankers['crossvalidated']=retrievers.get_class('tfidf')('crossvalidated')
KBSource={'stackoverflow','datascience','crossvalidated','AI'}
# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------
docDB=DBLoader.MongoStackExchange(host='10.1.1.9',port=50000)

sranker=retrievers.get_class('semantic')(programmingalpha.ModelPath+"/pytorch_model.bin")
fetchEach=25

def process(query, k=1):
    results=[]
    docs=[]
    for dbName in KBSource:
        ranker=rankers[dbName]
        doc_names, doc_scores = ranker.closest_docs(query, fetchEach)
        #print("found {}/{} in {}".format(len(doc_names),k,dbName))
        for i in range(len(doc_names)):
            results.append(
                {"Id":doc_names[i],
                 "score":doc_scores[i],
                 "db":dbName}
            )
            docDB.useDB(dbName)
            docs.append(
                {"Id":"{}|||{}".format(doc_names[i],dbName),
                 "text":docDB.get_doc_text("question",doc_names[i])
                 }
                 )


    results=heapq.nlargest(k,key=lambda doc:doc["score"],iterable=results)

    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score','Doc']
    )
    for i in range(len(results)):
        docDB.useDB(results[i]["db"])
        table.add_row([ i + 1, "{}-{}".format(results[i]["Id"],results[i]["db"]),
                        '%.5g' % results[i]["score"], docDB.get_doc_text("question",results[i]["Id"],0,0) ])
    print(table)


    logger.info("using semantic ranker to resort {} entries".format(len(docs)))
    sresults=sranker.closest_docs(query,docs,k)

    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score','Doc']
    )
    for i in range(len(sresults)):
        r=sresults[i]
        doc_id,dbName=r[0].split("|||")
        doc_id=int(doc_id)
        score=float(r[1])


        docDB.useDB(dbName)
        table.add_row([ i + 1, r[0], '%.5g' % score, docDB.get_doc_text(doc_id,0,0) ])

    print(table)

banner = """
Interactive Programming Alpha For AI Retriever
>> process(question, k=1)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
