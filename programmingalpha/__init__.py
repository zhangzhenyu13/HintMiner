import json

#bert model
BertBasePath="/home/LAB/zhangzy/ShareModels/uncased_L-12_H-768_A-12/"
BertLargePath="/home/LAB/zhangzy/ShareModels/uncased_L-24_H-1024_A-16/"
#openai GPT model
openAIGPTPath="/home/LAB/zhangzy/ShareModels/openAIGPT/"
#transformer-XL
transformerXL="/home/LAB/zhangzy/ShareModels/transformerXL/"
#gpt-2 model
GPT2Path="/home/LAB/zhangzy/ShareModels/GPT2/"

#global project path
ConfigPath="/home/LAB/zhangzy/ProgrammingAlpha/ConfigData/"
DataCases="/home/LAB/zhangzy/ProgrammingAlpha/dataCases/"

DataPath="/home/LAB/zhangzy/ProjectData/"
ModelPath="/home/LAB/zhangzy/ProjectModels/"

#embedding
Glove42="/home/LAB/zhangzy/ShareModels/Embeddings/glove.42B/"
Glove840="/home/LAB/zhangzy/ShareModels/Embeddings/glove.840B/"
Glove6="/home/LAB/zhangzy/ShareModels/Embeddings/glove.6B/"
GloveStack="/home/LAB/zhangzy/ShareModels/Embeddings/stackexchange/"
Bert768="/home/LAB/zhangzy/ShareModels/Embeddings/bert/"
openAI768="/home/LAB/zhangzy/ShareModels/Embeddings/openAI/"



def loadConfig(filename):
    with open(filename,"r") as f:
        config=json.load(f)
    return config

def saveConfig(filename,config):
    with open(filename, "w") as f:
        json.dump(config,f)


