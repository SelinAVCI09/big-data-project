from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.streaming import StreamingContext
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import json
import joblib

# 1. SparkSession ve SparkContext Oluşturma
spark = SparkSession.builder \
    .appName("Retail Anomaly Detection") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

sc = spark.sparkContext

# 2. Veri Hazırlama
def load_data(file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    df = df.filter((col("Quantity") > 0) & (col("UnitPrice") > 0)).na.drop()
    df = df.withColumn("TotalPrice", col("Quantity") * col("UnitPrice"))

    assembler = VectorAssembler(inputCols=["Quantity", "UnitPrice", "TotalPrice"], outputCol="features")
    df = assembler.transform(df)

    df = df.withColumn("label", when((col("Quantity") > 10) & (col("UnitPrice") > 5), 1).otherwise(0))
    return df

# 3. Random Forest Modeli Eğitimi
def train_random_forest(train_data):
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, seed=42)
    return rf.fit(train_data)

# 4. Autoencoder Modeli Eğitimi
def train_autoencoder(data_np):
    input_dim = data_np.shape[1]

    autoencoder = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='sigmoid')
    ])

    autoencoder.compile(optimizer=Adam(), loss='mse')
    autoencoder.fit(data_np, data_np, epochs=10, batch_size=32, verbose=1)

    autoencoder.save('autoencoder_model.h5')
    return autoencoder

# 5. Kafka Entegrasyonu
def process_stream(rf_model, autoencoder_model):
    ssc = StreamingContext(sc, 5)

    def process_rdd(rdd):
        if not rdd.isEmpty():
            records = rdd.map(lambda x: json.loads(x))
            records_df = spark.read.json(records)

            assembler = VectorAssembler(inputCols=["Quantity", "UnitPrice", "TotalPrice"], outputCol="features")
            records_df = assembler.transform(records_df)

            rf_predictions = rf_model.transform(records_df)

            features_np = np.array(records_df.select("features").rdd.map(lambda row: row[0].toArray()).collect())
            reconstruction = autoencoder_model.predict(features_np)
            mse = np.mean(np.power(features_np - reconstruction, 2), axis=1)
            anomaly = (mse > 0.01).astype(int)

            results = [{
                "features": features.tolist(),
                "rf_prediction": int(rf_pred),
                "autoencoder_anomaly": int(anomaly[idx])
            } for idx, (features, rf_pred) in enumerate(zip(features_np, rf_predictions.select("prediction").collect()))]

            for result in results:
                print(json.dumps(result))

    kafka_stream = ssc.socketTextStream("localhost", 9092)
    kafka_stream.foreachRDD(process_rdd)

    ssc.start()
    ssc.awaitTermination()

# 6. Main İşlevi
if __name__ == "__main__":
    file_path = "data/onlineretail.csv"
    data = load_data(file_path)

    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

    print("Training Random Forest...")
    rf_model = train_random_forest(train_data)

    print("Training Autoencoder...")
    feature_np = np.array(train_data.select("features").rdd.map(lambda row: row[0].toArray()).collect())
    autoencoder_model = train_autoencoder(feature_np)

    print("Evaluating Random Forest...")
    predictions = rf_model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Random Forest Accuracy: {accuracy:.2f}")

    print("Starting Kafka Stream Processing...")
    process_stream(rf_model, autoencoder_model)
