from confluent_kafka import Consumer, KafkaException, KafkaError
import json

# Kafka consumer kurulumu
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'retail_group',
    'auto.offset.reset': 'earliest'
})

consumer.subscribe(['retail_sales'])

try:
    while True:
        msg = consumer.poll(timeout=1.0)
        
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print('%% Reached end of partition.')
            elif msg.error():
                raise KafkaException(msg.error())
        else:
            record = json.loads(msg.value().decode('utf-8'))
            print(f"Received: {record}")

except KeyboardInterrupt:
    print("Consumer stopped manually.")
finally:
    consumer.close()
