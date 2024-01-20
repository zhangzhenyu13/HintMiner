from programmingalpha.DataSet.DBLoader import MongoStackExchange,MongoWikiDoc,MongodbAuth
import argparse
import regex as re


def initPool():
    global db
    db=MongoStackExchange(**MongodbAuth)

def splitTags():
    db.useDB("crossvalidated")

    batch_q=[]
    if "tmpQuestions" in db.stackdb.list_collection_names():
        db.stackdb.drop_collection("tmpQuestions")


    questions=db.stackdb.create_collection("tmpQuestions")

    for q in db.questions.find():
        tags=q["Tags"]
        if type(tags)!=list:
            q["Tags"]=tag_matcher.findall(tags)

        batch_q.append(q)
        if len(batch_q)%args.batch_size==0:
            questions.insert_many(batch_q)
            batch_q.clear()
            print("insert a batch,current size {}/{}".format(questions.count(),db.questions.count()))

    if len(batch_q)>0:
        questions.insert_many(batch_q)
        batch_q.clear()
        print("insert a batch,current size {}/{}".format(questions.count(),db.questions.count()))

    name=db.questions.name
    db.stackdb.drop_collection(db.questions.name)
    questions.rename(name)

def createkeysForWikiDocs():
    db=MongoWikiDoc(**MongodbAuth)

    db.useDB("wikidocs")
    articles=db.wikidb.get_collection('articles')
    tmp_collection=db.wikidb.create_collection('tmp_articles')
    cache_docs=[]
    for doc in articles.find().batch_size(args.batch_size):
        newdoc={"Id":int(doc['id']),"text":doc["text"],"Title":doc["title"],"url":doc["url"]}
        cache_docs.append(newdoc)
        if len(cache_docs)%args.batch_size==0:
            tmp_collection.insert_many(cache_docs)
            print("inserted %d docs"%tmp_collection.count())
            cache_docs.clear()

    if len(cache_docs)>0:
        tmp_collection.insert_many(cache_docs)
        print("inserted %d docs"%tmp_collection.count())

    tmp_collection.create_index("Id")


def distinctMoveWiki(src,dst):
    db=MongoWikiDoc(**MongodbAuth)
    db.useDB("wikidocs")
    id_set=set()
    dst_collection=db.wikidb.create_collection(dst)
    src_colletion=db.wikidb.get_collection(src)
    insert_cache=[]
    print("count before distinct",src_colletion.count())

    for doc in src_colletion.find().batch_size(10000):
        if doc["Id"] in id_set:
            continue
        id_set.add(doc["Id"])
        insert_cache.append(doc)
        if len(insert_cache)%10000==0:
            dst_collection.insert_many(insert_cache)
            insert_cache.clear()
    if len(insert_cache)>0:
        dst_collection.insert_many(insert_cache)

    dst_collection.create_index("Id")

    print("count after distinct",dst_collection.count())

def distinctMoveStack(src,dst):
    db=MongoStackExchange("10.1.1.9","36666")
    db.useDB("stackoverflow")
    id_set=set()

    dst_collection=db.stackdb[dst]
    src_collection=db.stackdb[src]

    insert_cache=[]
    print("count before distinct",src_collection.count())

    for doc in src_collection.find().batch_size(10000):
        if doc["Id"] in id_set:
            continue
        id_set.add(doc["Id"])
        insert_cache.append(doc)
        if len(insert_cache)%10000==0:
            dst_collection.insert_many(insert_cache)
            insert_cache.clear()
            print("process {}/{}".format(dst_collection.count(),src_collection.count()))
    if len(insert_cache)>0:
        dst_collection.insert_many(insert_cache)
        print("process {}/{}".format(dst_collection.count(),src_collection.count()))


    print("count after distinct",dst_collection.count())

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mongodb', type=str, help='mongodb host, e.g. mongodb://10.1.1.9:27017/',default='mongodb://10.1.1.9:27017/')
    parser.add_argument('--batch_size', type=str, default=10000)
    args=parser.parse_args()

    initPool()
    tag_matcher=re.compile(r"<.*?>",re.I)

    #createkeysForWikiDocs()
    #splitTags()

    #distinctMoveWiki("categories","tmp_tags")
    distinctMoveStack("questions_tmp","questions_unique")
