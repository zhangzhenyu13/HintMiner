import prettytable
import logging
from programmingalpha.retrievers.tfidf_doc_ranker import TfidfDocRanker
from programmingalpha.retrievers.relation_searcher import RelationSearcher
from programmingalpha import retrievers

from programmingalpha.DataSet import DBLoader
import programmingalpha

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


logger.info('Initializing ranker...')


rankers={}
rankers['stackoverflow']=TfidfDocRanker('stackoverflow')
rankers['AI']=TfidfDocRanker('AI')
rankers['datascience']=TfidfDocRanker('datascience')
rankers['crossvalidated']=TfidfDocRanker('crossvalidated')
KBSource={'stackoverflow','datascience','crossvalidated','AI'}
# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------
docDB=DBLoader.MongoStackExchange(**DBLoader.MongodbAuth)

sranker=RelationSearcher(programmingalpha.ModelPath + "/pytorch_model.bin")
fetchEach=10

def process(query, k=1):

    docs=[]
    logger.info("Retieving corpus using ngram retrieval model")
    for dbName in KBSource:
        ranker=rankers[dbName]
        doc_names, doc_scores = ranker.closest_docs(query, fetchEach)

        for i in range(len(doc_names)):

            docDB.useDB(dbName)
            docDB.setDocCollection(retrievers.WorkingDocCollection)
            docs.append(
                {"Id":"{}|||{}".format(doc_names[i],dbName),
                 "text":docDB.get_doc_text(doc_names[i],chunk_answer=0)
                 }
                 )

    logger.info("Using semantic ranker model to resort {} entries".format(len(docs)))
    results=sranker.closest_docs(query,docs,k)

    table = []
    for i in range(len(results)):

        r=results[i]
        doc_id,dbName=r[0].split("|||")
        doc_id=int(doc_id)

        docDB.useDB(dbName)
        docDB.setDocCollection(retrievers.WorkingDocCollection)
        table.append("<p>"+docDB.get_doc_text(doc_id,0,10)+"</p>")

        print(i+1,r[1])

    #print(table)

    return " <br/>\n".join(table)


