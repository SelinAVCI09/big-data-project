import pandas as pd
from confluent_kafka import Producer
import json

# Kafka Producer kurulumu
producer = Producer({'bootstrap.servers': 'localhost:9092'})

# Veri dosyasını yükleme
data = pd.read_csv('/Users/selinavci/Desktop/retail-anomaly-detection/data/OnlineRetail.csv', encoding='ISO-8859-1')

# Mesaj gönderme raporu
def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

# Kafka'ya veriyi gönderme
for index, row in data.iterrows():
    message = {
        'InvoiceNo': row['InvoiceNo'],
        'StockCode': row['StockCode'],
        'Description': row['Description'],
        'Quantity': row['Quantity'],
        'InvoiceDate': row['InvoiceDate'],
        'UnitPrice': row['UnitPrice'],
        'CustomerID': row['CustomerID'],
        'Country': row['Country']
    }
    producer.produce(
        topic='retail_sales',
        key=str(index),
        value=json.dumps(message),
        callback=delivery_report
    )
    producer.poll(0)

producer.flush()
