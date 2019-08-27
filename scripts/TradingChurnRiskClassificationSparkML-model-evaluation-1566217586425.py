#!/usr/bin/python

import pandas as pd
import json
from uuid import uuid4
import time, sys, os, shutil, glob, io, requests
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, Model, PipelineModel
from pyspark.sql import SQLContext
import dsx_core_utils
from dsx_ml.ml import save_evaluation_metrics


# setup dsxr environmental vars from command line input
from dsx_ml.ml import dsxr_setup_environment
dsxr_setup_environment()

# define variables
args = {"dataset": "/datasets/TradingCustomerSparkMLEval.csv", "published": "false", "threshold": {"metric": "f1Score", "min_value": 0.7, "mid_value": 0.87}, "evaluator_type": "multiclass", "execution_type": "DSX", "remoteHost": "", "remoteHostImage": "", "livyVersion": "livyspark2"}
model_path = os.path.join(os.getenv("DSX_PROJECT_DIR"), "models", os.getenv("DEF_DSX_MODEL_NAME", "TradingChurnRiskClassificationSparkML"), os.getenv("DEF_DSX_MODEL_VERSION", "1"), "model")

# create spark context
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# load the input data

input_data = os.getenv("DEF_DSX_DATASOURCE_INPUT_FILE", os.getenv("DSX_PROJECT_DIR") + args.get("dataset"))
dataframe = SQLContext(sc).read.csv(input_data , header="true", inferSchema = "true")

# load the model from disk 
model_rf = PipelineModel.load(model_path)


startTime = int(time.time())

# generate predictions
predictions = model_rf.transform(dataframe)

threshold = {'metric': 'f1Score', 'min_value': 0.7, 'mid_value': 0.87}

# replace "label" below with the numeric representation of
# the label column that you defined while training the model
labelCol = "label"

# create evaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol=labelCol)

# compute evaluations
eval_fields = {
        "accuracyScore": evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}),
        "f1Score": evaluator.evaluate(predictions, {evaluator.metricName: "f1"}),
        "weightedPrecisionScore": evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"}),
        "weightedRecallScore":  evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"}),
        "thresholdMetric": threshold["metric"],
        "thresholdMinValue": threshold["min_value"],
        "thresholdMidValue": threshold["mid_value"]
    }

# feel free to customize to your own performance logic using the values of "good", "poor", and "fair".
if(eval_fields[eval_fields["thresholdMetric"]] >= threshold.get('mid_value', 0.70)):
    eval_fields["performance"] = "good"
elif(eval_fields[eval_fields["thresholdMetric"]] <= threshold.get('min_value', 0.25)):
    eval_fields["performance"] = "poor"
else:
    eval_fields["performance"] = "fair"

save_evaluation_metrics(eval_fields, "TradingChurnRiskClassificationSparkML", "1", startTime)