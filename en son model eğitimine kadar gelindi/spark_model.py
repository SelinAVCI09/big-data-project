from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.streaming import StreamingContext
from confluent_kafka import Producer, Consumer, KafkaError
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import json
import joblib
from sklearn.metrics import f1_score

# 1. SparkSession ve SparkContext Oluşturma
spark = SparkSession.builder \
    .appName("Retail Anomaly Detection") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

sc = spark.sparkContext

# 2. Veri Hazırlama
def load_data(file_path):
    # CSV dosyasını Spark DataFrame olarak oku
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Negatif değerleri temizle ve eksik verileri kaldır
    df = df.filter((col("Quantity") > 0) & (col("UnitPrice") > 0)) \
           .na.drop(subset=["Quantity", "UnitPrice"])
    
    # Toplam fiyat özelliğini ekle
    df = df.withColumn("TotalPrice", col("Quantity") * col("UnitPrice"))
    
    # Özellikleri vektör haline getirme
    feature_columns = ["Quantity", "UnitPrice", "TotalPrice"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)
    
    # Hedef sütunu ekle (örnek veriler için rastgele hedef)
    df = df.withColumn("label", when((col("Quantity") > 10) & (col("UnitPrice") > 5), 1).otherwise(0))
    
    return df

# 3. Random Forest Modeli Eğitimi
def train_random_forest(train_data):
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, seed=42)
    rf_model = rf.fit(train_data)
    return rf_model

# 4. Autoencoder Modeli Eğitimi
def train_autoencoder(data_rdd):
    # RDD'den numpy array'e dönüştür
    data_np = np.array(data_rdd.collect())
    
    # Veriyi ölçeklendir
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(data_rdd.toDF())  # DataFrame'e dönüştürme işlemi
    data_scaled = scaler_model.transform(data_rdd.toDF())  # DataFrame'e dönüştürüp ölçekleme
    
    # Autoencoder modeli oluştur ve eğit
    autoencoder = Sequential([
        Dense(64, activation='relu', input_shape=(data_scaled.select("scaled_features").head()[0].size,)),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(data_scaled.select("scaled_features").head()[0].size, activation='sigmoid')
    ])
    autoencoder.compile(optimizer=Adam(), loss='mse')
    autoencoder.fit(data_scaled.select("scaled_features").rdd.map(lambda x: np.array(x[0])).collect(),
                    data_scaled.select("scaled_features").rdd.map(lambda x: np.array(x[0])).collect(),
                    epochs=10, batch_size=32, verbose=1)
    
    # Model ve scaler'ı kaydet
    autoencoder.save('autoencoder_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    
    return autoencoder, scaler

# 5. Kafka Entegrasyonu
def process_stream(rf_model, autoencoder_model, scaler):
    # Kafka Producer oluştur
    producer = Producer({'bootstrap.servers': 'localhost:9092'})
    
    # Spark Streaming için StreamingContext oluştur
    ssc = StreamingContext(sc, 5)  # 5 saniyelik mini batch
    kafka_stream = ssc.socketTextStream("localhost", 9092)  # Kafka'dan veri al
    
    def process_rdd(rdd):
        if not rdd.isEmpty():
            records = rdd.map(lambda x: json.loads(x))  # Kafka'dan gelen JSON mesajını çöz
            records_df = spark.read.json(records)  # Spark DataFrame'e dönüştür
            
            # Özellikleri vektörize et
            assembler = VectorAssembler(inputCols=["Quantity", "UnitPrice", "TotalPrice"], outputCol="features")
            records_df = assembler.transform(records_df)
            
            # Random Forest tahmini yap
            rf_predictions = rf_model.transform(records_df)
            rf_results = rf_predictions.select("features", "prediction")
            
            # Autoencoder ile anomali tespiti yap
            records_rdd = records_df.select("features").rdd.map(lambda row: row.features.toArray())
            data_np = np.array(records_rdd.collect())
            data_scaled = scaler.transform(data_np)
            reconstruction = autoencoder_model.predict(data_scaled)
            mse = np.mean(np.power(data_scaled - reconstruction, 2), axis=1)
            anomaly = (mse > 0.01).astype(int)  # Örnek eşik değeri
            
            # Sonuçları Kafka'ya yaz
            for idx, row in enumerate(rf_results.collect()):
                result = {
                    "features": row.features.tolist(),
                    "rf_prediction": int(row.prediction),
                    "autoencoder_anomaly": int(anomaly[idx])
                }
                output_topic = "normal_data" if result["rf_prediction"] == 0 and result["autoencoder_anomaly"] == 0 else "anomaly_data"
                producer.produce(output_topic, value=json.dumps(result))
                producer.flush()
    
    kafka_stream.foreachRDD(process_rdd)
    ssc.start()
    ssc.awaitTermination()

# 6. Main İşlevi
if __name__ == "__main__":
    # Veriyi yükle
    file_path = "data/onlineretail.csv"
    data = load_data(file_path)
    
    # Veri setini böl
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    # Random Forest Modelini Eğit
    print("Training Random Forest...")
    rf_model = train_random_forest(train_data)
    
    # Autoencoder Modelini Eğit
    print("Training Autoencoder...")
    feature_rdd = train_data.select("features").rdd.map(lambda row: row.features.toArray())
    autoencoder_model, scaler = train_autoencoder(feature_rdd)
    
    # Test Sonuçları
    print("Evaluating Random Forest...")
    predictions = rf_model.transform(test_data)
    
    # F1 Skoru Hesaplama
    y_true = test_data.select("label").rdd.map(lambda row: row.label).collect()
    y_pred = predictions.select("prediction").rdd.map(lambda row: row.prediction).collect()
    f1 = f1_score(y_true, y_pred)
    print(f"Random Forest F1 Score: {f1:.2f}")
    
    # Accuracy Hesaplama
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Random Forest Accuracy: {accuracy:.2f}")
    
    # Kafka Entegrasyonunu Başlat
    print("Starting Kafka Stream Processing...")
    process_stream(rf_model, autoencoder_model, scaler)
