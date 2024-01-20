from programmingalpha.DataSet.DBLoader import MongoStackExchange


batch_size=10000

def initDB(dbName):

    db=MongoStackExchange(host='10.1.1.9',port=50000)
    db.useDB(dbName)

    return db

def traverseQuestions(db:MongoStackExchange):
    print("traverse questions")
    n_questions=db.questions.count()
    questions=db.questions.find().batch_size(batch_size)
    questionDict={}
    for q in questions:
        q_id=q["Id"]
        if 'AcceptedAnswerId' not in q or q["AcceptedAnswerId"]=='':
            accepted_id=-1
        else:
            accepted_id=q["AcceptedAnswerId"]
        questionDict[q_id]={
            "AcceptedAnswerId":accepted_id,
            "AnswerCount":q["AnswerCount"],
            "Answers":set(),

        }


        if len(questionDict)%batch_size==0:
            print("progress:{}/{}".format(len(questionDict),n_questions))

    print("progress:{}/{}".format(len(questionDict),n_questions))

    return questionDict

def traverseAnswers(db:MongoStackExchange,questionDict:dict):
    print("traverse answers")
    n_answers=db.answers.count()
    answers=db.answers.find().batch_size(batch_size)
    count=0
    for ans in answers:
        if ans["ParentId"] in questionDict.keys():
            ans_id=ans["Id"]
            q_id=ans["ParentId"]
            questionDict[q_id]["Answers"].add(ans_id)
            count+=1

        if count%batch_size==0:
            print("progress:{}/{}".format(count,n_answers))

    print("progress:{}/{}".format(count,n_answers))

def updateDB(db:MongoStackExchange,data:dict,collectionName="QAIndexer"):
    print("insert data into",collectionName)
    if collectionName in db.stackdb.list_collection_names():
        db.stackdb.drop_collection(collectionName)
    newCollection=db.stackdb.create_collection(collectionName)

    cache=[]
    count=0
    for k,v in data.items():
        doc=v
        doc["Id"]=k
        doc["Answers"]=list(doc["Answers"])
        cache.append(doc)

        if len(cache)%batch_size==0:
            count+=len(cache)
            newCollection.insert_many(cache)
            cache.clear()
            print("progress:{}/{}".format(count,len(data)))

    if len(cache)>0:
        count+=len(cache)
        newCollection.insert_many(cache)
        print("progress:{}/{}".format(count,len(data)))

if __name__ == '__main__':
    db=initDB("stackoverflow")
    questionDict=traverseQuestions(db)
    traverseAnswers(db,questionDict)
    updateDB(db,questionDict)
