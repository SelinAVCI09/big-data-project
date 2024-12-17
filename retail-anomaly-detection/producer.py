import pandas as pd
from confluent_kafka import Producer
import json

# Kafka producer kurulumu
producer = Producer({
    'bootstrap.servers': 'localhost:9092'
})

# Veri dosyasını yükle (farklı bir encoding ile)
data = pd.read_csv('/Users/selinavci/Desktop/retail-anomaly-detection/data/OnlineRetail.csv', encoding='ISO-8859-1')

# Mesaj gönderme fonksiyonu
def delivery_report(err, msg):
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

# Veriyi Kafka'ya gönder
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
    producer.produce('retail_sales', key=str(index), value=json.dumps(message), callback=delivery_report)
    producer.poll(0)

# Producer'ı kapat
producer.flush(10)
