from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
import logging
import argparse
import tqdm
import json
import multiprocessing
from programmingalpha.tokenizers import BertTokenizer
from programmingalpha.Utility.TextPreprocessing import InformationAbstrator

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)

def init(questionsData_G):


    global tokenizer,textExtractor
    never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]","[NUM]","[CODE]")
    tokenizer=BertTokenizer(programmingalpha.ModelPath+"knowledgeSearcher/vocab.txt",never_split=never_split)
    textExtractor=InformationAbstrator(args.maxLength,tokenizer)

    filter_funcs={
        "pagerank":textExtractor.page_rank_texts,
        "lexrankS":textExtractor.lexrankSummary,
        "klS":textExtractor.klSummary,
        "lsaS":textExtractor.lsarankSummary,
        "textrankS":textExtractor.textrankSummary,
        "reductionS":textExtractor.reductionSummary
    }
    textExtractor.initParagraphFilter(filter_funcs[args.extractor])

    global questionsData
    questionsData=questionsData_G.copy()
    logger.info("process {} init".format(multiprocessing.current_process()))


def fetchQuestionData(q_ids_set):
    questionsData={}
    query={
        "$or":[
            {"FavoriteCount":{"$gte":1}},
         ]
    }

    db=MongoStackExchange(host='10.1.1.9',port=50000)
    db.useDB(args.db)
    for question in tqdm.tqdm(db.questions.find(query).batch_size(args.batch_size),desc="loading questions"):

        Id=question["Id"]
        if Id not in q_ids_set:
            continue
        del question["_id"]
        questionsData[Id]=question

    logger.info("loaded: questions({})".format(len(questionsData)))

    return questionsData


def _genCore(link):

    label=link["label"]
    q1,q2=link["pair"]

    if not (q1 in questionsData and q2 in questionsData):
        #if label=='duplicate':
        #    print(link,q1 in questionsData, q2 in questionsData)
        return None

    question1=questionsData[q1]
    question2=questionsData[q2]

    title1=" ".join(textExtractor.tokenizer.tokenize(question1["Title"]))
    title2=" ".join(textExtractor.tokenizer.tokenize(question2["Title"]))

    body1=question1["Body"]
    body2=question2["Body"]

    body1=textExtractor.clipText(body1)
    body2=textExtractor.clipText(body2)


    question1=[title1]+body1
    question2=[title2]+body2

    record={"id_1":q1,"id_2":q2,"label":label,"q1":question1,"q2":question2}

    return record

def generateQuestionCorpus(labelData,questionsDataGlobal):

    cache=[]
    batch_size=args.batch_size

    batches=[labelData[i:i+batch_size] for i in range(0,len(labelData),batch_size)]

    workers=multiprocessing.Pool(args.workers,initializer=init,initargs=(questionsDataGlobal,))

    counter={'unrelated': 0, 'direct': 0, 'transitive': 0, 'duplicate': 0}

    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-inference.json","w") as f:
        for batch_labels in tqdm.tqdm(batches,desc="processing documents"):
            for record in workers.map(_genCore,batch_labels):
                if record is not None:
                    counter[record["label"]]+=1
                    #cache.append(record)
                    cache.append(json.dumps(record)+"\n")

            f.writelines(cache)
            cache.clear()

        workers.close()
        workers.join()

    logger.info("after extratcing informatiev paragraphs: {}".format(counter))

def main():

    labelData=[]
    q_ids_set=set()
    with open(programmingalpha.DataPath+"/linkData/"+args.db.lower()+"-labelPair.json","r") as f:
        for line in f:
            record=json.loads(line)
            labelData.append(record)
            q_ids_set.update(record["pair"])


    questionsDataGlobal=fetchQuestionData(q_ids_set)

    q_ids_set=questionsDataGlobal.keys()
    labelDataNew=[]
    for ld in labelData:
        id1,id2=ld["pair"]
        if id1 not in q_ids_set or id2 not in q_ids_set:
            continue
        labelDataNew.append(ld)

    labels=map(lambda ll:ll["label"],labelData)

    import collections
    logger.info(collections.Counter(labels))

    generateQuestionCorpus(labelDataNew,questionsDataGlobal)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--db', type=str, default="AI")
    parser.add_argument('--maxLength', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--extractor', type=str, default="lexrankS")

    args = parser.parse_args()

    logger.info("task db is {}".format(args.db))

    main()
