import requests
import json


class SearcherBase(object):
    def __init__(self,host,port):
        self.host=host
        self.port=port

    def _getPostUrl(self,dbName,collectionName):
        post_url= "http://{}:{}/{}/{}/_search".format(self.host,self.port,dbName,collectionName)
        return post_url

class SearchStackExchange(SearcherBase):
    def __init__(self,host,port):
        SearcherBase.__init__(self=self,host=host,port=port)

    def retriveSimilarQuestions(self,query_str,dbName,size=10):
        post_url=self._getPostUrl(dbName,"questions")
        postData={
            "query": {
                "multi_match" : {
                    "query" : query_str,
                    "fields" : ["Title","Body"]
                    }
                },
            "size": size
        }
        postData=json.dumps(postData)
        response=requests.post(post_url,postData)

        results=json.loads(response.text)
        if "hits" in results:
            results=results["hits"]
        else:
            return []
        if "hits" in results:
            results=results["hits"]
        else:
            return []

        questions=[]
        for question in results:
            question=question["_source"]
            txt=question["Title"]+"\n"+question["Body"]

            questions.append({"Id":question["Id"],"content":txt})

        return questions

    def getAnswers(self,question_id,dbName,size=1):
        post_url=self._getPostUrl(dbName,"answers")
        postData={

                "query": {
                    "multi_match" : {
                        "query" : question_id,
                        "fields" : ["ParentId"]
                    }
                }
                ,"size":size
                , "sort":[
                  {
                    "Score": {
                      "order": "desc"
                    }
                  }
                ]
        }
        postData=json.dumps(postData)
        response=requests.post(post_url,postData)
        results=json.loads(response.text)

        if "hits" in results:
            results=results["hits"]
        else:
            return []
        if "hits" in results:
            results=results["hits"]
        else:
            return []

        answers=[]
        for ans in results:
            ans=ans["_source"]
            answers.append({"Id":ans["Id"],"content":ans["Body"]})

        return answers

class SearchWiki(SearcherBase):
    def __init__(self,host,port):
        SearcherBase.__init__(self,host,port)

    def retriveRelativePages(self,query_str="",size=1):
        post_url=self._getPostUrl("wikipedia","pages")
        postData={
          "size":size,
          "query": {
            "bool": {
              "should": [
                {"match": {
                  "text": query_str
                }}
              ]
            }
          }
        }
        postData=json.dumps(postData)
        response=requests.post(post_url,postData)

        results=json.loads(response.text)
        if "hits" in results:
            results=results["hits"]
        else:
            return []
        if "hits" in results:
            results=results["hits"]
        else:
            return []

        pages=[]
        for page in results:
            page=page["_source"]
            txt=page["text"]

            pages.append({"Id":page["id"],"content":txt})

        return pages

if __name__ == '__main__':
    size=5
    searcher=SearchStackExchange(host='10.1.1.9',port='9200')
    questions=searcher.retriveSimilarQuestions(query_str="what is transformer?",dbName='crossvalidated',size=size)
    q_ids=[]
    for q in questions:
        q_id=q["Id"]
        content=q["content"]
        q_ids.append(q_id)
        print(content)
        print("-"*50)
    print("+"*60)
    for q_id in q_ids:
        answers=searcher.getAnswers(question_id=q_id,dbName="crossvalidated",size=size)
        size-=1
        print(len(answers),"answers for",q_id)
        for ans in answers:
            print(ans)
        print("="*50)

    wikiSearcher=SearchWiki(host='10.1.1.9',port='9200')
    pages=wikiSearcher.retriveRelativePages("what is transformer?",size=5)
    for p in pages:
        print(p)
        print("*"*50)
