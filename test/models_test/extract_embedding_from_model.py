from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import torch,numpy as np
import programmingalpha
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling_openai import OpenAIGPTModel
from pytorch_pretrained_bert.modeling_transfo_xl import TransfoXLModel
from programmingalpha.tokenizers import BertTokenizer,OpenAIGPTTokenizer,TransfoXLTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


modelPath={
    "bert": programmingalpha.BertBasePath,
    "openAI": programmingalpha.openAIGPTPath
}

def extractBert():
    model = BertModel.from_pretrained(modelPath[args.model])
    #print(model);exit(10)
    embeddings=model.embeddings.word_embeddings
    print(embeddings.num_embeddings)
    print(embeddings.weight.size())
    weight=embeddings.weight.detach().numpy()
    tokenizer=BertTokenizer.from_pretrained(modelPath[args.model])
    #print(tokenizer.ids_to_tokens)
    #for i in range(10):
    #    print(weight[i])
    with open(programmingalpha.Bert768+"embeddings.txt","w") as f:
        vec_strs=[]
        for i  in range(len(weight)):
            vec=weight[i]
            vec_str=list(map(lambda x :str(x),vec))
            token=tokenizer.ids_to_tokens[i]
            vec_str.insert(0,token)
            vec_str=" ".join(vec_str)
            vec_strs.append(vec_str+"\n")

        def turnIndexs(index1,index2):
            tmp=vec_strs[index1]
            vec_strs[index1]=vec_strs[index2]
            vec_strs[index2]=tmp
        turnIndexs(0,1)
        turnIndexs(0,100)
        f.writelines(vec_strs)




def extractOpenAI():
    model = OpenAIGPTModel.from_pretrained(modelPath[args.model])
    embeddings=model.tokens_embed
    print(embeddings.num_embeddings)
    print(embeddings.weight.size())
    tokenizer=OpenAIGPTTokenizer.from_pretrained(modelPath[args.model])
    weight=embeddings.weight.detach().numpy()
    #print(tokenizer.decoder)
    with open(programmingalpha.openAI768+"embeddings.txt","w") as f:
        for i  in range(len(weight)):
            vec=weight[i]
            vec_str=list(map(lambda x :str(x),vec))
            token=tokenizer.decoder[i]
            vec_str.insert(0,token)
            vec_str=" ".join(vec_str)
            f.writelines(vec_str+"\n")


    with open(programmingalpha.openAI768+"vocab.txt","w") as f:
        for i in range(len(weight)):
            token=tokenizer.decoder[i]
            f.write(token+"\n")


def main():

    if args.model not in modelPath:
        args.model="bert"



    if args.model == "openAI":
        extractOpenAI()
    else:
        extractBert()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model", default="bert", type=str,
                        help="defaut is bert")

    args = parser.parse_args()

    main()
