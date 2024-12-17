import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from confluent_kafka import Consumer, KafkaException, KafkaError
import json

# Kafka consumer kurulumu
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'retail_group',
    'auto.offset.reset': 'earliest'
})

consumer.subscribe(['retail_sales'])

# Kafka'dan veri çekmek için bir fonksiyon
def get_data_from_kafka(consumer, num_records=100):
    data = []
    count = 0
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
                data.append(record)
                count += 1
                if count >= num_records:
                    break
    except KeyboardInterrupt:
        print("Consumer stopped manually.")
    finally:
        consumer.close()
    return pd.DataFrame(data)

# Veri Türlerini Birleştirme Fonksiyonu
def unify_column_types(data, column_name, target_type=str):
    try:
        data[column_name] = data[column_name].astype(target_type)
        print(f"Successfully converted {column_name} to {target_type}.")
    except Exception as e:
        print(f"Error converting {column_name} to {target_type}: {e}")
    return data

# Eksik Veri Kontrolü ve Temizleme Fonksiyonu
def handle_missing_values(data, method='drop', fill_value=None):
    if method == 'drop':
        data = data.dropna()
        print("Dropped missing values.")
    elif method == 'fill':
        data = data.fillna(fill_value)
        print(f"Filled missing values with {fill_value}.")
    return data

# Görselleştirme Fonksiyonu
def visualize_data(data, x_column, y_column, plot_type='bar'):
    plt.figure(figsize=(10, 6))
    if plot_type == 'bar':
        sns.barplot(x=x_column, y=y_column, data=data, errorbar=None)
    elif plot_type == 'scatter':
        sns.scatterplot(x=x_column, y=y_column, data=data)
    elif plot_type == 'line':
        sns.lineplot(x=x_column, y=y_column, data=data)
    plt.title(f"{plot_type.capitalize()} Plot of {y_column} vs {x_column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Ana Fonksiyon
def main():
    # Kafka'dan veri çekiyoruz
    print("Kafka'dan veri çekiliyor...")
    data = get_data_from_kafka(consumer, num_records=100)

    print("Orijinal Veri:")
    print(data.head())  # Veri çıktısını görmek için ilk 5 satırını gösteriyoruz

    # Eksik verileri temizleme
    data = handle_missing_values(data, method='fill', fill_value=0)

    # Veri türlerini düzenleme
    data = unify_column_types(data, 'InvoiceNo', target_type=str)

    # Anomaly sütunu ekleyerek sınıflandırma problemi oluşturma
    # Örnek: Fiyatı 3'ten düşük ve miktarı 2'den az olanları "anomalili" olarak işaretle
    data['Anomaly'] = ((data['Price'] < 3) & (data['Quantity'] < 2)).astype(int)

    print("\nSınıflandırma için İşlenmiş Veri:")
    print(data)

    # Özellikler (X) ve hedef değişkeni (y) ayırma
    X = data[['Quantity', 'Price']]  # Özellikler: Quantity ve Price
    y = data['Anomaly']             # Hedef: Anomaly (0 = normal, 1 = anomalili)

    # Eğitim ve test veri setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Logistic Regression modeli oluşturma ve eğitme
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test seti tahmini
    y_pred = model.predict(X_test)

    # Model performansı değerlendirme
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Performansı:")
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Görselleştirme
    print("\nGörselleştirme: StockCode'e göre Quantity (Bar Grafiği)")
    visualize_data(data, x_column='StockCode', y_column='Quantity', plot_type='bar')

    print("\nGörselleştirme: StockCode'e göre Price (Çizgi Grafiği)")
    visualize_data(data, x_column='StockCode', y_column='Price', plot_type='line')

    print("\nGörselleştirme: Quantity'e göre Price (Scatter Grafiği)")
    visualize_data(data, x_column='Quantity', y_column='Price', plot_type='scatter')

if __name__ == "__main__":
    main()
