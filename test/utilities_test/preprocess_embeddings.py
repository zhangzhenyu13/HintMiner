import argparse
import tqdm

infile="../../testdata/vectors.txt"#programmingalpha.GloveStack+"vocab.txt"
outputfile=infile+"-transformed"
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--db', type=str, default="datascience")
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--mode', type=str, default="vectors",help="vectors or vocabs")
parser.add_argument("--dimension",type=int,default=50)

args=parser.parse_args()

if args.input:
    infile=args.input
if args.output:
    outputfile=args.output

with open(infile,"r") as f:
    lines=f.readlines()

    if args.mode=="vocab":
        with open(outputfile,"w") as f2:
            cache=[]
            for line in tqdm.tqdm(lines):
                ss=" ".join(line.split()[:-1])+"\n"
                cache.append(ss)

            f2.writelines(cache)

    elif args.mode=="vectors":
        vecs=[]
        for line in lines:
            gvec=" ".join(line.split()[-args.dimension:])+"\n"
            vecs.append(gvec)

        with open(outputfile,"w") as f:
            f.writelines(vecs)

    else:
        raise ValueError("mode not defined: {}".format(args.mode))



