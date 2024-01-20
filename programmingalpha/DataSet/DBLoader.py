import pymongo
from urllib.parse import quote_plus
from programmingalpha.Utility.TextPreprocessing import PreprocessPostContent


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MongoDbConnector(object):
    def __init__(self,host,port,user=None,passwd=None):
        if user is not None:
            url = "mongodb://{}:{}@{}:{}".format (
                quote_plus(user), quote_plus(passwd), host,port)
        else:
            url="mongodb://{}:{}".format(host,port)

        self.client=pymongo.MongoClient(url)

    def close(self):
        self.client.close()

    def useDB(self,dbName):
        if dbName is None or dbName not in self.client.list_database_names():
            raise KeyError("{} is not found".format(dbName))


class DocDB(object):
    @staticmethod
    def filterNILStr(s):
        filterFunc=lambda s: s and s.strip()
        return list(filter(filterFunc,s.split('\n')))

    #for doc retrival interface
    def initData(self):
        raise NotImplementedError

    def __setDocCollection(self,collectionName):
        raise NotImplementedError

    def get_doc_text(self,collectionName, doc_id, **kwargs):
        raise NotImplementedError

    def get_doc_ids(self,collectionName):
        raise NotImplementedError


class MongoStackExchange(MongoDbConnector,DocDB):
    textExtractor=PreprocessPostContent()
    def __init__(self,host,port,user=None,passwd=None):

        MongoDbConnector.__init__(self,host,port,user,passwd)

    def initData(self):
        self.questions=self.stackdb["questions"]
        self.answers=self.stackdb["answers"]

    def useDB(self,dbName):
        super(MongoStackExchange,self).useDB(dbName)
        self.stackdb=self.client[dbName]
        self.initData()
        return True

    #for doc retrival interface
    def __setDocCollection(self,collectionName):
        self.docs=self.stackdb[collectionName]

    def get_doc_text(self, collectionName, doc_id, is_question=True, chunk_size = 20):
        self.__setDocCollection(collectionName)
        doc_json=self.docs.find_one({"Id":doc_id})

        if doc_json is None:
            logger.info("error found none in db:{}, doc_id:{}".format(self.stackdb.name,self.docs.name,doc_id))
            return None

        if is_question:
            doc="\n".join([doc_json["Title"],doc_json["Body"]])
        else:
            doc=doc_json["Body"]


        if chunk_size>0:
            doc=self.textExtractor.getPlainTxt(doc)
            doc='\n'.join(self.filterNILStr(doc)[:chunk_size])

        return doc

    def get_doc_ids(self,collectionsName):
        self.__setDocCollection(collectionsName)
        doc_ids=[]
        for doc in self.docs.find():
            doc_ids.append(doc["Id"])
        return doc_ids




    def getBatchAcceptedQIds(self,batch_size=1000,query=None):
        # generator for question ids with accpted answers
        batch=[]

        if not query:
            results=self.questions.find().batch_size(batch_size)
        else:
            results=self.questions.find(query).batch_size(batch_size)

        for x in results:
            #print(x)
            if 'AcceptedAnswerId' not in x or x["AcceptedAnswerId"]=='':
                continue

            batch.append(x)
            if len(batch)%batch_size==0:
                yield batch
                batch.clear()

        if len(batch)>0:
            yield batch

    def searchForAnswers(self,question_id):
        ans=self.answers.find_one({"ParentId":question_id})
        return ans


class MongoWikiDoc(MongoDbConnector,DocDB):
    def __init__(self,host,port,user=None,passwd=None):
        MongoDbConnector.__init__(self,host,port,user,passwd)

    def useDB(self,dbName="wikipedia"):
        super(MongoWikiDoc, self).useDB(dbName)

        self.wikidb=self.client[dbName]
        self.initData()

    def initData(self):
        self.pages=self.wikidb["pages"]
        #self.tags=self.wikidb[""]

    def __setDocCollection(self,collectionName):
        self.docs=self.wikidb[collectionName]

    def get_doc_text(self,collectionName, doc_id, chunk_size=5):
        self.__setDocCollection(collectionName)
        doc_json=self.docs.find_one({"id":doc_id})

        doc=doc_json["text"]

        if chunk_size>0:
            doc=self.filterNILStr(doc)[1:chunk_size]
            doc='\n'.join(doc)


        return doc

    def get_doc_ids(self,collectionName):
        self.__setDocCollection(collectionName)
        doc_ids=[]
        for doc in self.docs.find().batch_size(10000):
            doc_ids.append(doc["id"])
        return doc_ids


if __name__ == '__main__':
    db=MongoStackExchange(host='10.1.1.9',port='36666')
    db.useDB("stackoverflow")
    print(db.questions.count())

    text=db.get_doc_text(collectionName="questions",doc_id=1083,chunk_size=10)
    print(text)
