def shuffleLinesFile(file):
    import random
    with open(file,"r") as f, open(file+"-random","w") as f2:
        lines=f.readlines()
        random.shuffle(lines)
        f2.writelines(lines)


def _recoverSent(texts,tokenizer=None):
    text=" ".join(texts)
    if tokenizer is None:
        text=" ".join(text.split())
    else:
        text=" ".join(tokenizer.tokenize(text))
    return text

def _constructSrc(record):
    question=_recoverSent(record["question"])
    context=_recoverSent(record["context"])
    seq_src=[]
    question_tokens=question.split()[:150]
    context_tokens=context.split()[:1200]

    seq_src+=question_tokens
    seq_src+=["[SEP]"]
    seq_src+=context_tokens
    seq_src+=["[SEP]"]

    return " ".join(seq_src)+"\n"
def buildUnsolvedSrc(file):
    import json
    with open(file,"r") as f, open(file+'-seq2seq',"w") as f2:
        lines=f.readlines()
        records=map(lambda r:json.loads(r.strip()),lines)
        records=map(_constructSrc,records)
        f2.writelines(records)


if __name__ == '__main__':
    #test4()
    #testBertService()
    #testSummarize()
    file="/home/LAB/zhangzy/ProjectData/Corpus/unsolved"
    buildUnsolvedSrc(file)
    pass
