from programmingalpha.Utility.metrics import LanguageMetrics
import numpy as np
import argparse
import programmingalpha

def computeScore(summary,refs):

    #print("sum=>",summary)
    #print("refs=>",refs)
    rouge_1=lan_metric.rouge_1_score(summary,refs)
    rouge_2=lan_metric.rouge_2_score(summary,refs)
    rouge_l=lan_metric.rouge_l_score(summary,refs)
    rouge_be=lan_metric.rouge_be_score(summary,refs)
    bleu = lan_metric.rouge_be_score(summary,refs)

    metric_score={"rouge-1":rouge_1,"rouge-2":rouge_2,"rouge-l":rouge_l,"rouge-be":rouge_be,"bleu":bleu}

    return metric_score

def loadRefAndSum(sum_file,ref_file):
    with open(sum_file,"r") as f1, open(ref_file,"r") as f2:
        summaries=f1.readlines()
        refs=f2.readlines()
    return zip(summaries,refs)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    maxLen=50
    parser.add_argument("--reference_file",type=str,default=programmingalpha.DataPath+"seq2seq/valid-dst")
    parser.add_argument("--summary_file",type=str,default=programmingalpha.DataPath+"predictions/predict-%d.txt"%maxLen)
    args=parser.parse_args()
    lan_metric=LanguageMetrics()
    scores=[]
    for summary, ref in loadRefAndSum(args.summary_file,args.reference_file):
        scores.append(computeScore(summary,ref))

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
