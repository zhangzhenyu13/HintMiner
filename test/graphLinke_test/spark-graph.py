from __future__ import print_function

import sys,os
from pyspark.sql import SparkSession
import graphframes
from pyspark.sql.types import Row,StructField,StructType,StringType
os.environ["PYSPARK_PYTHON"]="/home/LAB/zhangzy/anaconda3/bin/python"

def parseEdges(link):
    #link=json.loads(link)

    id_a,id_b=link["PostId"],link["RelatedPostId"]
    r=link["LinkTypeId"]
    if r==3:
        w=0
    elif r==1:
        w=1
    else:
        raise ValueError("unexpected value {} for link type".format(r))
    return (id_a,id_b,w)

def changeToDF_v1(rdd,table_view):
    def f(x):
        d={}
        for i in range(len(x)):
            d[str(i)]=x[i]
        return d
    newDF=rdd.map(lambda x:Row(**f(x))).toDF()
    newDF.createOrReplaceTempView(table_view)
    return newDF

def changeToDF_v2(rdd,schemaString,table_view):
    fields = map( lambda fieldName : StructField(fieldName, StringType(), nullable = False), schemaString.split(" "))
    fields=list(fields)
    schema=StructType(fields)
    if len(fields)==3:
        rowRdd=rdd.map(lambda x:Row(x[0],x[1],x[2])) # 3 for edges
    elif len(fields)==1:
        rowRdd=rdd.map(lambda x:Row(x)) # 1 for nodes
    else:
        raise ValueError("not defined fields in this program")
    newDF=spark.createDataFrame(rowRdd,schema)
    newDF.createOrReplaceTempView(table_view)
    return newDF

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("PostLinksRelation")\
        .getOrCreate()
    #people = spark.createDataFrame([("Bilbo Baggins",  50), ("Gandalf", 1000), ("Thorin", 195), ("Balin", 178), ("Kili", 77),
    #        ("Dwalin", 169), ("Oin", 167), ("Gloin", 158), ("Fili", 82), ("Bombur", None)], ["name", "age"])
    #people.write.format("com.mongodb.spark.sql.DefaultSource").mode("append").save()
    #'''

    #df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
    df=spark.read.json(sys.argv[1])
    df.printSchema()
    edges=df.rdd.map(parseEdges).cache()
    nodes=edges.map(lambda x:(x[0],x[1])).flatMap(lambda X:X).distinct()
    edges=changeToDF_v2(edges,"src dst weight","edges")
    #nodes=changeToDF_v1(nodes,"nodes")
    nodes=changeToDF_v2(nodes,"id","nodes")
    g=graphframes.GraphFrame(nodes,edges)
    g.dropIsolatedVertices()

    print("+"*100)
    vertices=g.vertices.collect()
    vertices=list(map(lambda x:x["id"],vertices))
    print(vertices[:36])
    print("+"*100)


    distances_data=None
    for i in range(len(vertices)):
        print("*"*50+"{}/{}".format(i+1,len(vertices))+"*"*50)
        for j in range(i):
            distances=g.bfs(
                fromExpr="id!=-123",
                toExpr="id!=-123",
                #edgeFilter="weight!=0",
                maxPathLength=2
            )
            if distances_data is not None and distances is not None and distances.count()>0:
                distances_data=distances_data.union(distances)
            else:
                distances_data=distances


    distances.write.json(sys.argv[1]+"-output.json")
    #results = g.pageRank(resetProbability=0.01, maxIter=20)
    #results.vertices.select("id", "pagerank").show()
    #'''
    spark.stop()
