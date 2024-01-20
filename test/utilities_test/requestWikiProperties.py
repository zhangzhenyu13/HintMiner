from programmingalpha.DataSet.DBLoader import MongoWikiDoc
from multiprocessing.dummy import Pool as ThreadPool
from programmingalpha.Utility.WebCrawler import AgentProxyCrawler
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

db=MongoWikiDoc(host='10.1.1.9',port=36666)
db.useDB('wikidocs')

crawled_ids=db.get_doc_ids('categories')

wikidoc_ids=db.get_doc_ids('articles')

crawler=AgentProxyCrawler()

def requestData(id=None):

    url="https://en.wikipedia.org/w/api.php"
    params={
        "action":"query",
        "format":"json",
        "pageids":id,
        #"titles":"Gradient descent",
        "prop":"categories"
    }

    R=crawler.get(url=url,params=params,timeout=3)

    data=R.json()
    cats=[]

    try:
      #  print(data)
        data=data["query"]["pages"][str(id)]
        for cat in data["categories"]:
     #       print(cat)
            cats.append(cat["title"][9:])
    except KeyError as e:
        #e.with_traceback()

        logger.info("data content",data)
        #exit(1)

    if 'title' not in data:
        return None

    return {"Id":id,"Title":data["title"],"categories":cats}


#cats=requestData(238493);print(cats);exit(10)


logger.info("init with {} doc ids and {} crawled ids".format(len(wikidoc_ids),len(crawled_ids)))
wikidoc_ids=set(wikidoc_ids).difference(crawled_ids)
wikidoc_ids=list(wikidoc_ids)
logger.info("incrementally crawl {} ids".format(len(wikidoc_ids)))
batch_size=100

batch_doc_ids=[wikidoc_ids[i:i+batch_size] for i in range(0,len(wikidoc_ids),batch_size)]

workers=ThreadPool(batch_size)
collection_cats=db.wikidb["categories"]
cache_insterion=[]
for i in range(len(batch_doc_ids)):
    logger.info(25*"*"+"requesting batches {}/{}".format(i+1,len(batch_doc_ids))+"*"*25)
    batch_results=workers.map(requestData,batch_doc_ids[i])
    cache_insterion.extend(batch_results)
    if len(cache_insterion)>batch_size:
        cache_insterion=list(filter(None,cache_insterion))
        collection_cats.insert_many(cache_insterion)
        cache_insterion.clear()
workers.close()
workers.join()


