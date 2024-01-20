from programmingalpha.tokenizers import BertTokenizer,OpenAIGPTTokenizer,TransfoXLTokenizer
import programmingalpha
import argparse
import os

tokenizers={
    "bert",
    "openAI",
    "transfoXL"
}

def init():
    global tokenizer
    if args.tokenizer == "openAI":
        tokenizer=OpenAIGPTTokenizer(programmingalpha.openAIGPTPath+"/openai-gpt-vocab.json",
                                     programmingalpha.openAIGPTPath+"/openai-gpt-merges.txt")
    elif args.tokenizer == "transfoXL":
        tokenizer=TransfoXLTokenizer()
    else:
        tokenizer=BertTokenizer(os.path.join(programmingalpha.BertBasePath,"vocab.txt"))


def tokenize(text):
    text=text[0]
    #print("trying to tokenize=>",text)
    tokenized_text=tokenizer.tokenize(text)

    return " ".join(tokenized_text)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--tokenizer",type=str,default="")
    parser.add_argument("--file",type=str,default="")

    args=parser.parse_args()

    if args.tokenizer not in tokenizers:
        args.tokenizer="bert"

    inputfile=args.file
    outputfile=inputfile+"-tokenized-"+args.tokenizer

    init()
    from pyspark.sql import SparkSession
    spark = SparkSession\
        .builder\
        .appName("tokenize text with "+args.tokenizer)\
        .getOrCreate()

    tokenized=spark.read.text(inputfile).rdd.map(tokenize)

    tokenized.saveAsTextFile(outputfile)
    spark.stop()
