#for glove vectors clearning

import programmingalpha
import argparse
import tqdm
infile="../../testdata/vectors.txt"#programmingalpha.GloveStack+"vocab.txt"
outputfile=infile+"-transformed"
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--db', type=str, default="datascience")
parser.add_argument('--input', type=str, default=infile)
parser.add_argument('--output', type=str, default=outputfile)
args=parser.parse_args()

def _convert(token):
        if token == '-LRB-':
            return '('
        if token == '-RRB-':
            return ')'
        if token == '-LSB-':
            return '['
        if token == '-RSB-':
            return ']'
        if token == '-LCB-':
            return '{'
        if token == '-RCB-':
            return '}'
        return token

with open(infile,"r") as f:
    with open(outputfile,"w") as f2:
        lines=f.readlines()
        cache=[]
        for line in tqdm.tqdm(lines):
            ss=line.split()
            tok=ss[0]
            others=ss[1:]
            ss=" ".join([_convert(tok)," ".join(others)])+"\n"
            cache.append(ss)

            if len(cache)%args.batch_size==0:
                f2.writelines(cache)
                cache.clear()
        if len(cache)>0:
            f2.writelines(cache)

