# including reuired libraries
import os
import sys

from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def clean_data(df):
    # cleaning header 
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

    

"""main function for application"""
if __name__ == "__main__":
    
    # Create spark application
    spark = SparkSession.builder \
        .appName('nithya_cs643_wine_prediction') \
        .getOrCreate()

    # create spark context to report logging information related spark
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    # Load and parse the data file into an RDD of LabeledPoint.
    if len(sys.argv) > 3:
        sys.exit(-1)
    elif len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if not("/" in input_path):
            input_path = "data/csv/" + input_path
        model_path="/code/data/model/testdata.model"
        print("----Input file for test data is---")
        print(input_path)
    else:
        current_dir = os.getcwd() 
        print("-----------------------")
        print(current_dir)
        input_path = os.path.join(current_dir, "data/csv/testdata.csv")
        model_path= os.path.join(current_dir, "data/model/testdata.model")

    # read csv file in DataFram 
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(input_path))
    
    df1 = clean_data(df)
    # Split the data into training and test sets (30% held out for testing)
    # removing column not adding much value to prediction
    # removed 'residual sugar','free sulfur dioxide',  'pH',
    all_features = ['fixed acidity',
                        'volatile acidity',
                        'citric acid',
                        'chlorides',
                        'total sulfur dioxide',
                        'density',
                        'sulphates',
                        'alcohol',
                    ]
    
   
   
    rf = PipelineModel.load(model_path)
    
    preds = rf.transform(df1)
    print(preds.show(10))
    results = preds.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(
                                            labelCol='label', 
                                            predictionCol='prediction', 
                                            metricName='accuracy')

    # printing accuracy of above model
    accuracy = evaluator.evaluate(preds)
    print('Accuracy of my wine prediction model = ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('F1 score (Weighted) = ', metrics.weightedFMeasure())
    sys.exit(0)
