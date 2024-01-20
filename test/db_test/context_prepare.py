from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
from programmingalpha.Utility.TextPreprocessing import PreprocessPostContent
import json
import logging
import argparse
import tqdm
import multiprocessing

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



def init(questionsData_G,answersData_G,indexData_G,copy=True):
    global preprocessor
    preprocessor=PreprocessPostContent()

    global questionsData,answersData,indexData
    if copy:
        questionsData=questionsData_G.copy()
        answersData=answersData_G.copy()
        indexData=indexData_G.copy()
    else:
        questionsData=questionsData_G
        answersData=answersData_G
        indexData=indexData_G

    logger.info("process {} init".format(multiprocessing.current_process()))



def fetchQuestionData(q_ids_set):
    questionsData={}

    needed_answerIds=set()

    query={
        "$or":[
            {"AcceptedAnswerId":{"$exists":True,"$ne":''},"FavoriteCount":{"$gte":3}},
            {"AnswerCount":{"$gte":args.answerNum}},
         ]
    }

    for question in tqdm.tqdm(docDB.questions.find(query).batch_size(args.batch_size),desc="loading questions"):


        Id=question["Id"]

        if Id not in q_ids_set:
            continue

        del question["_id"]
        questionsData[Id]={"Title":question["Title"],"Body":question["Body"],"AcceptedAnswerId":question["AcceptedAnswerId"]}

        needed_answerIds.add(question["AcceptedAnswerId"])

    logger.info("loaded: questions({})".format(len(questionsData)))

    return questionsData, needed_answerIds

def fetchAnswerData(ansIdxGlobal,questionsDataGlobal):
    answersData={}

    for ans in tqdm.tqdm(docDB.answers.find().batch_size(args.batch_size),desc="loading answers"):

        Id=ans["Id"]

        if  Id not in ansIdxGlobal or ans["ParentId"] not in questionsDataGlobal:
            continue


        answersData[Id]={"Body":ans["Body"],"Score":ans["Score"]}

    logger.info("loaded: answers({})".format(len(answersData)))

    return answersData

def fetchIndexData(questionDataGlobal):
    indexData={}
    for indexer in tqdm.tqdm(docDB.stackdb["QAIndexer"].find().batch_size(args.batch_size),desc="loading indexers"):
        Id=indexer["Id"]

        if Id not in questionDataGlobal:
            continue

        del indexer["_id"]

        indexData[Id]=indexer

    logger.info("loaded: indexer({})".format(len(indexData)))

    return indexData

#generate Core
def _getBestAnswers(q_id,K):
    answers=[]
    if "AcceptedAnswerId" in questionsData[q_id]:
        ans_id=questionsData[q_id]["AcceptedAnswerId"]

        if ans_id in answersData:
            answer=answersData[ans_id]
            K-=1

    ans_idx=indexData[q_id]["Answers"]
    scored=[]
    for id in ans_idx:
        if id in answersData:
            scored.append((id,answersData[id]["Score"]))
    if scored:
        scored.sort(key=lambda x:x[1],reverse=True)
        for i in range(min(K-1,len(scored))):
            id=scored[i][0]
            answers.append(answersData[id])

    if K<args.answerNum:
        answers=[answer]+answers


    return answers

def _getPreprocess(txt):

    txt_processed=preprocessor.getPlainTxt(txt)

    if len(" ".join(txt_processed).split())<20:
        return None

    return txt_processed



def _genCore(distances):
    #try:
        q_id=distances["id"]

        #get question
        if q_id not in questionsData:
            return None

        question=questionsData[q_id]
        title=question["Title"]
        body=question["Body"]

        question =_getPreprocess(body)
        if not question:
            return None
        question=[title]+question

        #get answer
        answer=_getBestAnswers(q_id, K=args.answerNum)

        if not answer:
            return None

        answer=_getPreprocess(answer[0]["Body"])
        if not answer:
            return None


        #get context
        relative_q_ids=[]
        dists=distances["distances"]

        for id in dists:
            if id not in questionsData:
                continue

            if len(relative_q_ids)>=10:
                break

            if dists[id]==1:
                relative_q_ids.append(id)
            elif dists[id]==0:
                relative_q_ids.insert(0,id)
            else:
                pass

        if len(relative_q_ids)==0:
            return None



        context=[]
        for q_id in relative_q_ids:


            ans=_getBestAnswers(q_id,args.answerNum)
            if not ans:
                continue

            context.extend(ans)

        if len(context)==0:
            #logger.info("due to none context")
            return None

        context.sort(key=lambda ans:ans["Score"],reverse=True)
        contexts=[]
        for txt in context:
            txt=_getPreprocess(txt["Body"])
            if not txt:
                continue
            contexts.extend(txt)

        if len(contexts)==0:
            #logger.info("due to none context")
            return None


        record={"question":question,"context":contexts,"answer":answer}

        return record
    #except :
    #    logger.warning("except triggered for distance data: {}".format(distances))
    #    return None


def generateContextAnswerCorpusParallel(distanceData,questionsDataGlobal,answersDataGlobal,indexDataGlobal):

    cache=[]
    batch_size=args.batch_size
    batches=[distanceData[i:i+batch_size] for i in range(0,len(distanceData),batch_size)]

    workers=multiprocessing.Pool(args.workers,initializer=init,
                                 initargs=(questionsDataGlobal,answersDataGlobal,indexDataGlobal)
                                 )

    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-context.json","w") as f:
        for batch_links in tqdm.tqdm(batches,desc="processing documents"):

            for record in workers.map(_genCore,batch_links):
                if record is not None:

                    cache.append(json.dumps(record)+"\n")

            f.writelines(cache)
            cache.clear()


        workers.close()
        workers.join()

def generateContextAnswerCorpus(distanceData,questionsDataGlobal,answersDataGlobal,indexDataGlobal):

    cache=[]

    init(questionsDataGlobal,answersDataGlobal,indexDataGlobal,copy=False)

    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-context.json","w") as f:
        for link in tqdm.tqdm(distanceData,desc="processing documents"):
            record =_genCore(link)
            if record is not None:
                cache.append(json.dumps(record)+"\n")

            if len(cache)>args.batch_size:
                f.writelines(cache)
                cache.clear()

        if len(cache)>0:
            f.writelines(cache)
            cache.clear()

def main():
    logger.info("loading distance data")
    distance_file=programmingalpha.DataPath+"linkData/"+dbName.lower()+'-2graph.json'
    distance_data=[]
    q_ids_set=set()
    with open(distance_file,"r") as f:
        for line in f:
            path=json.loads(line)
            q_ids_set.add(path["id"])
            q_ids_set.update(path["distances"])
            distance_data.append(path)
    logger.info("loaded {} links data".format(len(distance_data)))


    questionsDataGlobal, ansIdxGlobal=fetchQuestionData(q_ids_set)
    answersDataGlobal=fetchAnswerData(ansIdxGlobal,questionsDataGlobal.keys())
    indexerDataGlobal=fetchIndexData(questionsDataGlobal.keys())

    distance_dataNew=[]
    for distance in distance_data:
        id=distance["id"]
        if len(distance["distances"])==0:
            continue

        if id not in questionsDataGlobal:
            continue

        new_distance={"id":int(id),"distances":{}}
        for k,v in distance["distances"].items():
            k=int(k)
            v=int(v)
            new_distance["distances"][k]=v

        distance_dataNew.append(new_distance)

    logger.info("finally loaded {} links data".format(len(distance_dataNew)))

    generateContextAnswerCorpusParallel(distance_dataNew,questionsDataGlobal,answersDataGlobal,indexerDataGlobal)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--db', type=str, default="crossvalidated")
    parser.add_argument('--lose_rate', type=float, default=0.5)
    parser.add_argument("--answerNum",type=int,default=5)

    parser.add_argument('--workers', type=int, default=32)

    args = parser.parse_args()

    docDB=MongoStackExchange(host='10.1.1.9',port=50000)
    dbName=args.db
    docDB.useDB(dbName)

    logger.info("processing db data: {}".format(dbName))

    main()
