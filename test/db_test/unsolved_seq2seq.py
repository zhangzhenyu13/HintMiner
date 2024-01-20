from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
from programmingalpha.tokenizers import BertTokenizer
from programmingalpha.Utility.TextPreprocessing import InformationAbstrator
import json
import logging
import argparse
import tqdm
import multiprocessing

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def init(questionsData_G,answersData_G,copy=True):
    global tokenizer,textExtractor
    never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]","[NUM]","[CODE]")
    tokenizer=BertTokenizer(programmingalpha.ModelPath+"knowledgeComprehension/vocab.txt",never_split=never_split)
    textExtractor=InformationAbstrator(500,tokenizer)

    filter_funcs={
        "pagerank":textExtractor.page_rank_texts,
        "lexrankS":textExtractor.lexrankSummary,
        "klS":textExtractor.klSummary,
        "lsaS":textExtractor.lsarankSummary,
        "textrankS":textExtractor.textrankSummary,
        "reductionS":textExtractor.reductionSummary
    }
    textExtractor.initParagraphFilter(filter_funcs[args.extractor])

    global questionsData,answersData
    if copy:
        questionsData=questionsData_G.copy()
        answersData=answersData_G.copy()
    else:
        questionsData=questionsData_G
        answersData=answersData_G

    logger.info("process {} init".format(multiprocessing.current_process()))


def fetchUnsolvedQuestionData(q_ids_set):
    questionsData={}


    query=[
        {
            "$match":{"$or":[
                {"AcceptedAnswerId":{"$exists":False} },
                {"AcceptedAnswerId":''}
            ] }
        },
        {
            "$match":{"$or":[
            {"FavoriteCount":{"$gte":3}},
            {"AnswerCount":{"$gte":5}},
            ] }
        }
        ,
        {"$sample": {"size": args.samples*5}}
    ]

    for question in tqdm.tqdm(docDB.questions.aggregate(pipeline=query,allowDiskUse=True),desc="loading unsolved questions"):


        Id=question["Id"]

        if Id not in q_ids_set:
            continue

        del question["_id"]
        questionsData[Id]={"Title":question["Title"],"Body":question["Body"]}


    logger.info("loaded: unsolved questions({})".format(len(questionsData)))

    return questionsData



def fetchQuestionData(q_ids_set):
    questionsData={}

    needed_answerIds=set()

    query=[
        {"$match":{
            "Id":{"$in":list(q_ids_set)}
            }
         },

        {
        "$match":{
            "$or":[
                {"AcceptedAnswerId":{"$exists":True,"$ne":''},"FavoriteCount":{"$gte":3}},
                {"AnswerCount":{"$gte":5}},
                 ]
            }
        }
    ]
    def _filerLowQualityQuestion(question):
        cond1=lambda q:("AcceptedAnswerId" not in q or not q["AcceptedAnswerId"])
        cond2=lambda q:("AnswerCount" not in q or not q["AnswerCount"] or int(q["AnswerCount"])<10)
        cond3=lambda q:("FavoriteCount" not in q or not q["FavoriteCount"] or int(q["FavoriteCount"])<3)
        if cond1(question) or ( cond2(question) and cond3(question) ):
            'high quality question only'
            return False

        return True

    for question in tqdm.tqdm(docDB.questions.aggregate(pipeline=query,allowDiskUse=True,batchSize=args.batch_size),desc="loading questions"):


        Id=question["Id"]

        if Id not in q_ids_set:
            continue

        if _filerLowQualityQuestion(question)==False:
            continue

        del question["_id"]
        questionsData[Id]={"Title":question["Title"],"Body":question["Body"],"AcceptedAnswerId":question["AcceptedAnswerId"]}

        needed_answerIds.add(question["AcceptedAnswerId"])

    logger.info("loaded: questions({})".format(len(questionsData)))

    return questionsData, needed_answerIds

def fetchAnswerData(ansIdxGlobal,questionsDataGlobal):
    answersData={}

    query={
        "ParentId":{"$in":list(questionsDataGlobal)}
    }
    for ans in tqdm.tqdm(docDB.answers.find(query).batch_size(args.batch_size),desc="loading answers"):

        Id=ans["Id"]

        if  Id not in ansIdxGlobal or ans["ParentId"] not in questionsDataGlobal:
            continue


        answersData[Id]={"Body":ans["Body"],"Score":ans["Score"]}

    logger.info("loaded: answers({})".format(len(answersData)))

    return answersData


#generate Core
def _getBestAnswer(q_id):
    if "AcceptedAnswerId" not in questionsData[q_id]:
        return None

    ans_id=questionsData[q_id]["AcceptedAnswerId"]

    if ans_id in answersData:
        answer=answersData[ans_id]
    else:
        return None

    return answer

def _getPreprocess(txt,maxLen,cal_lose=False):
    textExtractor.maxClip=maxLen
    if cal_lose:
        original=" ".join(textExtractor.processor.getPlainTxt(txt))
        before_len=len( original.split() )
        if before_len<5:
            #logger.info("bad zero:{}\n=>{}".format(txt,original))
            return "",0
    txt_processed=textExtractor.clipText(txt)

    if cal_lose:
        after_len=len(" ".join(txt_processed).split())
        lose_rate= after_len/before_len
        return txt_processed, lose_rate
    else:
        return txt_processed,None


def _genCore(distances):
    #try:
        q_id=distances["id"]
        question_id=q_id
        #get question
        if q_id not in questionsData:
            return None

        question=questionsData[q_id]
        title=" ".join( textExtractor.tokenizer.tokenize(question["Title"]) )
        body=question["Body"]

        question,_=_getPreprocess(body,args.questionLen)

        question=[title]+question


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


            ans=_getBestAnswer(q_id)
            if ans is None:
                continue
            ans_txt=ans["Body"]
            context.append(ans_txt)

        if len(context)==0:
            #logger.info("due to none context")
            return None

        context=" ".join(context)

        context,_=_getPreprocess(context,args.contextLen)

        if len(context)==0:
            #logger.info("due to none context")
            return None


        record={"Id":question_id,"source":args.db,"question":question,"context":context}
        #record=_constructSrc(record)
        record=json.dumps(record)+"\n"
        return record
    #except :
    #    logger.warning("except triggered for distance data: {}".format(distances))
    #    return None


def generateContextAnswerCorpusParallel(distanceData,questionsDataGlobal,answersDataGlobal):

    cache=[]
    batch_size=args.batch_size
    batches=[distanceData[i:i+batch_size] for i in range(0,len(distanceData),batch_size)]

    workers=multiprocessing.Pool(args.workers,initializer=init,
                                 initargs=(questionsDataGlobal,answersDataGlobal))

    with open(programmingalpha.DataPath+"Corpus/unsolved","a") as f:
        for batch_links in tqdm.tqdm(batches,desc="processing documents"):

            for record in workers.map(_genCore,batch_links):
                if record is not None:

                    cache.append(record)

            f.writelines(cache)
            cache.clear()


        workers.close()
        workers.join()


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


    unsolvedQuestionGlobal=fetchUnsolvedQuestionData(q_ids_set)
    distance_dataNew=[]
    needed_qids=set()
    for distance in distance_data:
        id=int(distance["id"])
        if len(distance["distances"])==0:
            continue

        if id not in unsolvedQuestionGlobal:
            continue

        new_distance={"id":id,"distances":{}}

        for k,v in distance["distances"].items():
            k=int(k)
            v=int(v)
            new_distance["distances"][k]=v
            needed_qids.add(k)

        distance_dataNew.append(new_distance)

    logger.info("finally loaded {} links data".format(len(distance_dataNew)))

    questionsDataGlobal, ansIdxGlobal=fetchQuestionData(needed_qids)
    answersDataGlobal=fetchAnswerData(ansIdxGlobal,questionsDataGlobal.keys())

    questionsDataGlobal.update(unsolvedQuestionGlobal)
    generateContextAnswerCorpusParallel(distance_dataNew,questionsDataGlobal,answersDataGlobal)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--samples', type=int, default=2000)

    parser.add_argument('--db', type=str, default="stackoverflow")
    parser.add_argument('--contextLen', type=int, default=1200)
    parser.add_argument('--questionLen', type=int, default=150)
    parser.add_argument('--lose_rate', type=float, default=0.5)

    parser.add_argument('--extractor', type=str, default="lexrankS")

    parser.add_argument('--workers', type=int, default=32)

    args = parser.parse_args()

    docDB=MongoStackExchange(host='10.1.1.9',port=50000)
    dbName=args.db
    docDB.useDB(dbName)

    main()
