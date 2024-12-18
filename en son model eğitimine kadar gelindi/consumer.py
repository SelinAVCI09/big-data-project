from confluent_kafka import Consumer, KafkaException, KafkaError, Producer
import json

# Kafka Consumer kurulumu
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'retail_group',
    'auto.offset.reset': 'earliest'
})

consumer.subscribe(['retail_sales'])

# Kafka Producer kurulumu
producer = Producer({
    'bootstrap.servers': 'localhost:9092',
    'queue.buffering.max.messages': 100000,  # Buffer limiti artırıldı
    'queue.buffering.max.kbytes': 1024 * 1024,  # 1 GB buffer
    'batch.num.messages': 1000,  # Batch sayısı artırıldı
    'linger.ms': 1000  # Mesajların belirli bir süre bekletilmesi
})

# Mesaj teslimat raporları için callback fonksiyonu
def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                raise KafkaException(msg.error())
        else:
            record = json.loads(msg.value().decode('utf-8'))
            # Negatif değerler için kontrol
            if record['Quantity'] < 0 or record['UnitPrice'] < 0:
                producer.produce(
                    topic='anomalies',
                    value=json.dumps(record),
                    callback=delivery_report  # Callback eklenerek başarılı teslimat takibi yapılır
                )
            else:
                producer.produce(
                    topic='normal_data',
                    value=json.dumps(record),
                    callback=delivery_report  # Callback eklenerek başarılı teslimat takibi yapılır
                )
            
            # Kafka producer tamponunu boşaltmak için poll çağırılır
            producer.poll(0)

            # Her 1000 mesajda bir flush işlemi yapılır
            producer.flush()

except KeyboardInterrupt:
    print("Consumer durduruldu.")
finally:
    # Kafka producer'ını düzgün şekilde kapatmak için flush çağrısı yapılır
    producer.flush()
    consumer.close()
