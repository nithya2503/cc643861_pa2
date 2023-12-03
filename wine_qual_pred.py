
"""
Nithya sri - nk659

To train the model and tune the model, spark application is being used
"""



'''
from module import function
module.function()

pyspark is an interface in python for writing spark application

To write the spark application, pyspark is used as an interface

Data is splitted into 3 parts, 2/3rd part of data is acquired for training, 1/3rd is for testing 
CrossValidator - will split dataset into dataset pairs. If we get 3 dataset pairs, 
each of 2/3 is used for training n 1/3 is used for testing model

The provided parameters in this grid are set to preset values by ParamGridBuilder.

Pipeline: guarantee that the characteristic phases of processing for training and test data are the same.

A set of column is combined into a single vector column using the VectorAssembler.

MulticlassMetrics is a multiple-class evaluator.

RandomForestClassifier - accommodates continuous and categorical characteristics, in addition to binary and multiclass labels.

An ML column of label indices is mapped from a string column of labels using a label indexer called a StringIndexer.

DataFrame creation, DataFrame registration as tables, SQL over tables, Cache tables, and more can all be done with SparkSession.



'''
import sys
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def prc_data(ch):
    # cleaning header 
    return ch.select(*(col(c).cast("double").alias(c.strip("\"")) for c in ch.columns))

    

#the main function begins here
if __name__ == "__main__":
    
    # spark application
    spk = SparkSession.builder \
        .appName('nithya_cs643_wine_prediction') \
        .getOrCreate()

  
    #to update log feed of spark, a spark context is created
    spark_context = spk.sparkContext
    spark_context.setLogLevel('ERROR')

    # Load and parse the data file into an RDD of LabeledPoint.
    if len(sys.argv) > 3:
        sys.exit(-1)
    elif len(sys.argv) == 3:
        inp_path = sys.argv[1]
        val_path = sys.argv[2]
        out_path = sys.argv[3] + "testmodel.model"
    else:
        inp_path = "s3://winepredbuck/TrainingDataset.csv"
        val_path = "s3://winepredbuck/ValidationDataset.csv"
        out_path="s3://winepredbuck/testmodel.model/"
    # reading the csvfile from dataframe 
    ch = (spk.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(inp_path))
    
    trn_dt_st = prc_data(ch)

    ch = (spk.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(val_path))
    
    valid_data_set = prc_data(ch)

#""fixed acidity"";""volatile acidity"";""citric acid"";""residual sugar"";""chlorides"";""free sulfur dioxide"";""total sulfur dioxide"";""density"";""pH"";""sulphates"";""alcohol"";""quality""
    
    features = ['fixed acidity',
                        'volatile acidity',
                        'citric acid',
                        'density',
                        'pH',
                        'sulphates',
                        'alcohol',
                        'residual sugar',
                        'chlorides',
                        'free sulfur dioxide',
                        'total sulfur dioxide',
                        'quality',
                    ]
    
    # VectorAssembler creating a single vector column name features using only all_features list columns 
    asmbler = VectorAssembler(inputCols=features, outputCol='features')
    
    # creating classification with given input values 
    indxr = StringIndexer(inputCol="quality", outputCol="label")

    # caching data so that it can be faster to use
    trn_dt_st.cache()
    valid_data_set.cache()
    
    # Choosing RandomForestClassifier for training
    rf = RandomForestClassifier(labelCol='label', 
                            featuresCol='features',
                            numTrees=150,
                            seed=149,
                            maxDepth=15,
                            maxBins=7, 
                            impurity='gini')
    
    # use this model to tune on training data
    pipeline = Pipeline(stages=[asmbler, indxr, rf])
    model = pipeline.fit(trn_dt_st)

    # validate the trained model on test data
    preds = model.transform(valid_data_set)

 
    rslts = preds.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(labelCol='label', 
                                        predictionCol='prediction', 
                                        metricName='accuracy')

    
    # printing accuracy of above model
    accuracy = evaluator.evaluate(preds)
    print('Test Accuracy of wine prediction model= ', accuracy)
    metrics = MulticlassMetrics(rslts.rdd.map(tuple))
    print('Weighted f1 score of wine prediction model = ', metrics.weightedFMeasure())

    
    # Retrain model on mutiple parameters 
    cvmodel = None
    paramGrid = ParamGridBuilder() \
            .addGrid(rf.maxBins, [9, 8, 4])\
            .addGrid(rf.maxDepth, [25, 6 , 9])\
            .addGrid(rf.numTrees, [500, 50, 150])\
            .addGrid(rf.minInstancesPerNode, [6])\
            .addGrid(rf.seed, [100, 200, 5043, 1000])\
            .addGrid(rf.impurity, ["entropy","gini"])\
            .build()
    pipeline = Pipeline(stages=[asmbler, indxr, rf])
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2)

  
    cvmodel = crossval.fit(trn_dt_st)
    
    #save the best model to new param `model` 
    model = cvmodel.bestModel
    print(model)
    # print accuracy of best model
    preds = model.transform(valid_data_set)
    rslts = preds.select(['prediction', 'label'])
    accuracy = evaluator.evaluate(preds)
    print('Accuracy of my wine_pred_model = ', accuracy)
    metrics = MulticlassMetrics(rslts.rdd.map(tuple))
    print('F1 score(weighted) of my wine_prediction_model = ', metrics.weightedFMeasure())

    # saving best model to s3
    model_path =out_path
    model.write().overwrite().save(model_path)
    sys.exit(0)