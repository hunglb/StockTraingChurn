#!/usr/bin/python

import sys, os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, Model, PipelineModel
from pyspark.sql import SQLContext
import pandas
import dsx_core_utils, re, jaydebeapi
from sqlalchemy import *
from sqlalchemy.types import String, Boolean


# setup dsxr environmental vars from command line input
from dsx_ml.ml import dsxr_setup_environment
dsxr_setup_environment()

# define variables
args = {'execution_type': 'DSX', 'target': '/datasets/batchscoreresult.csv', 'source': '/datasets/TradingCustomerSparkMLBatchScore.csv', 'output_type': 'Localfile', 'output_datasource_type': '', 'sysparm': '', 'remoteHost': '', 'remoteHostImage': '', 'livyVersion': 'livyspark2'}
input_data = os.getenv("DEF_DSX_DATASOURCE_INPUT_FILE", (os.getenv("DSX_PROJECT_DIR") + args.get("source")))
output_data = os.getenv("DEF_DSX_DATASOURCE_OUTPUT_FILE", (os.getenv("DSX_PROJECT_DIR") + args.get("target")))
model_path = os.getenv("DSX_PROJECT_DIR") + os.path.join("/models", os.getenv("DSX_MODEL_NAME","TradingChurnRiskClassificationSparkML"), os.getenv("DSX_MODEL_VERSION","1"),"model")

# create spark context
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# read test dataframe (inputJson = "input.json") 
testDF = SQLContext(sc).read.csv(input_data, header='true', inferSchema='true')

# load model
model_rf = PipelineModel.load(model_path)

# prediction
outputDF = model_rf.transform(testDF) 

# save scoring result to given target
scoring_df = outputDF.toPandas()

# save output to csv
scoring_df.to_csv(output_data, encoding='utf-8')