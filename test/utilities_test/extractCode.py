from programmingalpha.Utility.TextPreprocessing import PreprocessPostContent
import json

if __name__ == '__main__':
    from programmingalpha.DataSet.DBLoader import MongoStackExchange

    db=MongoStackExchange("mongodb://10.1.1.9")
    AIQA=db.stackdb["QAPForAI"]

    data=[]
    with open("testdata/quetions.txt","w") as f:
        for x in AIQA.find().batch_size(100):
            txt=x["question_title"]+" "+x["question_body"]

            processTxt=PreprocessPostContent()
            processTxt.raw_txt=txt
            result1,result2,result3=processTxt.getEmCodes(),processTxt.getCodeSnippets(),processTxt.getPlainTxt()

            data.append(json.dumps({"emcodes":result1,"snippets":result2,"plaintxt":result3})+"\n")

            if len(data)%1000==0:
                f.writelines(data)
                data.clear()

        if len(data)>0:
            f.writelines(data)
            data.clear()

            #print(txt)
            #print(len(result1),result1)
            #print(len(result2),result2)
            #print(len(result3),result3)
