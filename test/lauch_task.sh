#!/usr/bin/env bash
programCmd="app.py args..."

spark-submit
         --master yarn-cluster\
         --master local[*]\
	     #--num-executors 50\
	     #--executor-memory 8g\
		 #--executor-cores 2\
	     #--driver-memory 8g\
	     #--conf spark.driver.maxResultSize=4096\
	     #--conf spark.kryoserializer.buffer.max=2000\
		 #--conf spark.speculation=true \
		 #--conf spark.speculation.quantile=0.90\
	     $programCmd
