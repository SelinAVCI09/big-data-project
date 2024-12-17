from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# SparkSession oluşturma
spark = SparkSession.builder \
    .appName("Retail Anomaly Detection") \
    .getOrCreate()

# Kafka'dan veri okuyarak bir DataFrame oluşturma
kafka_bootstrap_servers = 'localhost:9092'
kafka_topic = 'retail_sales'

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic) \
    .load()

# Kafka mesajının değerini JSON formatında çözme
df = df.selectExpr("CAST(value AS STRING) as message")

# JSON verisini sütunlara ayırma
from pyspark.sql.functions import from_json, schema_of_json

# Mesajın şeması, Kafka'dan gelen JSON yapısına uygun olmalıdır
schema = schema_of_json('{"InvoiceNo": "string", "StockCode": "string", "Description": "string", "Quantity": "double", "InvoiceDate": "string", "UnitPrice": "double", "CustomerID": "string", "Country": "string"}')

df = df.select(from_json(col("message"), schema).alias("data")).select("data.*")

# Anomalileri bulma: Negatif Quantity veya UnitPrice değerlerini filtreleme
anomalies_df = df.filter((col("Quantity") < 0) | (col("UnitPrice") < 0))

# Anomalileri konsola yazdırma
query = anomalies_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
