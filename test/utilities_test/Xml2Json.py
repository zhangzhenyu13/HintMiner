import xml.sax as xmlsax
from xml.sax.saxutils import unescape
from xml.sax import ContentHandler as xmlContentHandler
import json
import argparse
import collections
import regex as re

class PostLinkHandler(xmlContentHandler):

    def __init__(self):
        xmlContentHandler.__init__(self)

        self.Id=None
        self.CreationDate=None
        self.PostId=None
        self.RelatedPostId=None
        self.LinkTypeId=None

        self.verboseSize=10000
        self.__cache=[]
        self.__ptr=0
        self.filename=None

    def startDocument(self):
        self.__ptr=0
        self.__file= open(file=self.filename,mode="w",encoding="utf-8")
        self.__cache.clear()

        self.labels=[]

    def endDocument(self):
        print("parsed {} post-links".format(self.__ptr))
        self.__file.writelines(self.__cache)
        self.__file.close()

        print("labels",collections.Counter(self.labels))

    def startElement(self,tag,attributes):

        if tag=="row":

            self.Id=int(attributes["Id"])
            self.CreationDate=attributes["CreationDate"]
            self.PostId=int(attributes["PostId"])
            self.RelatedPostId=int(attributes["RelatedPostId"])
            self.LinkTypeId=int(attributes["LinkTypeId"])
            self.__ptr+=1

            self.labels.append(self.LinkTypeId)
    def endElement(self,tag):
        if self.__ptr%self.verboseSize==0:
            print("parsed {} postLink".format(self.__ptr))
            self.__file.writelines(self.__cache)
            self.__cache.clear()

        if tag=="row":
            x={"Id":self.Id,"CreationDate":self.CreationDate,
               "PostId":self.PostId,"RelatedPostId":self.RelatedPostId,
               "LinkTypeId":self.LinkTypeId}
            self.__cache.append(json.dumps(x)+"\n")


#################################################
class PostHandler(xmlContentHandler):

    def __init__(self):
        xmlContentHandler.__init__(self)

        self.keys=[None,None]

        self.verboseSize=10000
        self.__cacheAns=[]
        self.__cacheQuest=[]
        self.__ptr=0
        self.path=None

        self.plainText=lambda text:", ".join(text.replace("<","").replace(">"," ").strip().split(" "))

    def startDocument(self):
        self.__ptr=0
        self.__fileAns= open(file=self.path+"/Answers.json",mode="w",encoding="utf-8")
        self.__fileQuest=open(file=self.path+"/Questions.json",mode="w",encoding="utf-8")

    def endDocument(self):
        print("parsed {} posts".format(self.__ptr))

        self.__fileAns.writelines(self.__cacheAns)
        self.__cacheAns.clear()

        self.__fileQuest.writelines(self.__cacheQuest)
        self.__cacheQuest.clear()

        self.__fileAns.close()
        self.__fileQuest.close()

        print(self.keys[0])
        print(self.keys[1])

    def startElement(self,tag,attributes):

        if tag=="row":
            x={}
            for attr in attributes.keys():
                try:
                    if "Date" in attr or "Title" in attr or "Body" in attr or "Tags" in attr or "Name" in attr:
                        x[attr]=unescape(attributes[attr])
                    else:
                        x[attr]=int(attributes[attr])

                except:
                    print(attr,attributes[attr])
                    exit(10)

            del x["PostTypeId"]
            if 1==int(attributes["PostTypeId"]):
                x["Tags"]=self.plainText(x["Tags"])
            x=json.dumps(x)+"\n"

            if 1==int(attributes["PostTypeId"]):
                self.__cacheQuest.append(x)
                if not self.keys[0]:
                    self.keys[0]=attributes.keys()

            elif 2==int(attributes["PostTypeId"]):

                self.__cacheAns.append(x)

                if not self.keys[1]:
                    self.keys[1]=attributes.keys()

            self.__ptr+=1

    def endElement(self,tag):
        if self.__ptr%self.verboseSize==0:
            print("parsed {} posts".format(self.__ptr))

        if len(self.__cacheAns)%self.verboseSize==0:
            self.__fileAns.writelines(self.__cacheAns)
            self.__cacheAns.clear()

        if len(self.__cacheQuest)%self.verboseSize==0:
            self.__fileQuest.writelines(self.__cacheQuest)
            self.__cacheQuest.clear()


#################################################
class TagHandler(xmlContentHandler):

    def __init__(self):
        xmlContentHandler.__init__(self)

        self.keys=None

        self.verboseSize=10000
        self.__cache=[]
        self.__ptr=0
        self.path=None

    def startDocument(self):
        self.__ptr=0
        self.__file= open(file=self.path+"/Tags.json",mode="w",encoding="utf-8")

    def endDocument(self):
        print("parsed {} tags".format(self.__ptr))

        self.__file.writelines(self.__cache)
        self.__cache.clear()

        self.__file.close()

        print(self.keys)

    def startElement(self,tag,attributes):

        if tag=="row":
            x={}
            for attr in attributes.keys():
                try:
                    if "Date" in attr or "Title" in attr or "Body" in attr or "Tags" in attr or "Name" in attr:
                        x[attr]=unescape(attributes[attr])
                    else:
                        x[attr]=int(attributes[attr])
                except:
                    print(attr,attributes[attr])
                    exit(10)
            x=json.dumps(x)+"\n"

            self.__cache.append(x)
            if not self.keys:
                self.keys=attributes.keys()

            self.__ptr+=1

    def endElement(self,tag):
        if self.__ptr%self.verboseSize==0:
            print("parsed {} tags".format(self.__ptr))

        if len(self.__cache)%self.verboseSize==0:
            self.__file.writelines(self.__cache)
            self.__cache.clear()




if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--xmlfile",type=str,help="input xml file to parse",default="../../testdata/posts.xml")
    parser.add_argument("--path",type=str,help="output dir for parsed results",default='../../testdata/')
    parser.add_argument("--method",type=int,help="1 for postlinks, 2 for posts, 3 for tags",default=2)

    args=parser.parse_args()
    xmlfile=args.xmlfile
    jsonfile=args.path+"/" +xmlfile[xmlfile.rfind("/")+1:xmlfile.rindex(".")]+".json"

    xmlParser=xmlsax.make_parser()
    xmlParser.setFeature(xmlsax.handler.feature_namespaces, 0)

    linkTypeHandler=PostLinkHandler()
    linkTypeHandler.filename=jsonfile


    postHandler=PostHandler()
    postHandler.path=args.path

    tagHandler=TagHandler()
    tagHandler.path=args.path

    handler={1:linkTypeHandler,2:postHandler, 3:tagHandler}

    xmlParser.setContentHandler(handler[args.method])

    xmlParser.parse(xmlfile)
