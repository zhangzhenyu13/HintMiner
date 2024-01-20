from pytorch_pretrained_bert.tokenization import BertTokenizer
import argparse
from programmingalpha.Utility.metrics import LanguageMetrics
from programmingalpha.Utility.TextPreprocessing import InformationAbstrator
import programmingalpha
from programmingalpha.DataSet.DBLoader import MongoStackExchange
import tqdm
import numpy as np

def init():
    global tokenizer,summaryFunc,lan_metric

    never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]","[NUM]","[CODE]")
    tokenizer=BertTokenizer(programmingalpha.ModelPath+"knowledgeComprehension/vocab.txt",never_split=never_split)
    textExtractor=InformationAbstrator(maxLen)
    lan_metric=LanguageMetrics()

    filter_funcs={
        "pagerank":textExtractor.page_rank_texts,
        "lexrankS":textExtractor.lexrankSummary,
        "klS":textExtractor.klSummary,
        "lsaS":textExtractor.lsarankSummary,
        "textrankS":textExtractor.textrankSummary,
        "reductionS":textExtractor.reductionSummary
    }
    summaryFunc=filter_funcs[args.extractor]

def tokenizeSentences(texts):

    tokenized_texts=[]
    for txt in texts:
        tokens=tokenizer.tokenize(txt)
        tokenized_texts.append(" ".join(tokens))

    return tokenized_texts

def computeScore(doc):
    answer=doc["answer"]
    context=doc["context"]
    answer=tokenizeSentences(answer)
    context=tokenizeSentences(context)

    refs=" ".join(answer)
    summary=summaryFunc(context)
    summary=" ".join(summary)
    summary=" ".join(summary.split()[:maxLen])


    #print("sum=>",summary)
    #print("refs=>",refs)
    rouge_1=lan_metric.rouge_1_score(summary,refs)
    rouge_2=lan_metric.rouge_2_score(summary,refs)
    rouge_l=lan_metric.rouge_l_score(summary,refs)
    rouge_be=lan_metric.rouge_be_score(summary,refs)
    bleu = lan_metric.rouge_be_score(summary,refs)

    metric_score={"rouge-1":rouge_1,"rouge-2":rouge_2,"rouge-l":rouge_l,"rouge-be":rouge_be,"bleu":bleu}

    return metric_score

def computeparallel(data_samples):
    scores=[]
    batches=[data_samples[i:i+args.batch_size] for i in range(0,len(data_samples),args.batch_size)]
    from multiprocessing import Pool as ProcessPool
    workers=ProcessPool(processes=args.workers,initializer=init)
    for batch in tqdm.tqdm(batches,desc="parallel computing score from context=>answer"):
        batch_scores=workers.map(computeScore,batch)
        scores.extend(batch_scores)
    workers.close()
    workers.join()
    return scores

def computeSingle(data_samples):
    scores=[]
    for doc in tqdm.tqdm(data_samples,desc="computing score from context=>answer"):

        metric_score=computeScore(doc)

        scores.append(metric_score)
    return scores

def fetchData():
    size=args.test_size
    collection=docDB.stackdb["context"]

    query=[
          {"$sample": {"size": size}}
        ]

    data_samples=collection.aggregate(pipeline=query,allowDiskUse=True,batchSize=args.batch_size)
    return data_samples

def summaryContext(data_samples):
    init()

    if args.workers<2:
        scores=computeSingle(data_samples)
    else:
        scores=computeparallel(data_samples)

    rouge_1=map(lambda x:x["rouge-1"],scores)
    rouge_2=map(lambda x:x["rouge-2"],scores)
    rouge_l=map(lambda x:x["rouge-l"],scores)
    rouge_be=map(lambda x:x["rouge-be"],scores)
    bleu=map(lambda x:x["bleu"],scores)

    rouge_1=np.mean(list(rouge_1))
    rouge_2=np.mean(list(rouge_2))
    rouge_l=np.mean(list(rouge_l))
    rouge_be=np.mean(list(rouge_be))
    bleu=np.mean(list(bleu))

    metric_score={"len":maxLen,"rogue-1":rouge_1,"rougue-2":rouge_2,"rougue-l":rouge_l,"rouge-be":rouge_be,"bleu":bleu}
    print(metric_score)
    return metric_score

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--test_size', type=int, default=2000)

    parser.add_argument('--db', type=str, default="corpus")

    parser.add_argument('--workers', type=int, default=1)

    args = parser.parse_args()

    docDB=MongoStackExchange(host='10.1.1.9',port=50000)
    dbName=args.db
    docDB.useDB(dbName)

    args=parser.parse_args()

    methods=["lexrankS","klS","lsaS","textrankS","reductionS"]
    data_samples=fetchData()
    data_samples=list(data_samples)

    all_scores=[]
    maxLen=50

    for summethod in methods:
        for length in range(50,550,50):

            args.extractor=summethod
            print("method ",args.extractor,"data size=",len(data_samples),"maxLen",length)
            maxLen=length
            metric=summaryContext(data_samples)
            all_scores.append(metric)
    print("*"*100)
    for ms in all_scores:
        print(ms)
