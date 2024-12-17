# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:12:04 2024

@author: pc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

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

# LSTM Modelini Eğitme ve Anomali Tespiti
def detect_anomalies_with_lstm(data, sequence_length=5):
    # `Quantity` ve `Price` sütunlarını kullanacağız
    series = data[['Quantity', 'Price']].copy()

    # Eksik değerleri sıfır ile doldur
    series.fillna(0, inplace=True)

    # Veriyi normalize et
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)

    # Zaman serisi verisini LSTM için hazırlama
    def create_sequence_data(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])
        return np.array(X), np.array(y)

    # Veriyi diziler halinde hazırlama
    X, y = create_sequence_data(scaled_series, sequence_length)
    
    # Eğitim ve test verilerini ayırma
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # LSTM Modelini oluşturma
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(sequence_length, X.shape[2])),
        Dense(2)  # Quantity ve Price çıktısı
    ])

    model.compile(optimizer=Adam(), loss='mean_squared_error')

    # Modeli eğitme
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Eğitim ve doğrulama kaybını görselleştirme
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Test verisinde tahmin yapma
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Gerçek ve tahmin edilen değerler arasındaki hata hesaplanması
    y_test_rescaled = scaler.inverse_transform(y_test)

    # MSE (Mean Squared Error) hesaplama
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    print(f"Test MSE: {mse}")

    # Anomalileri tespit etmek için hata eşiklerini belirleme
    residual = np.abs(y_test_rescaled - y_pred_rescaled)
    threshold = np.percentile(residual, 95)  # %95 hata aralığını kullanıyoruz
    anomalies = residual > threshold
    anomaly_count = np.sum(anomalies)
    
    print(f"Anomalies Detected: {anomaly_count}")

    # Görselleştirme
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled[:, 0], label='Gerçek Quantity')
    plt.plot(y_pred_rescaled[:, 0], label='Tahmin Edilen Quantity', linestyle='--')
    plt.title("Gerçek ve Tahmin Edilen Quantity Değeri")
    plt.legend()
    plt.show()

    return anomalies

# Ana Fonksiyon
def main():
    # Örnek veri seti
    data = pd.DataFrame({
        'InvoiceNo': [536365, '536366', 536367, '536368', None],
        'StockCode': ['85123A', '71053', '84406B', '84029G', '84406B'],
        'Quantity': [6, 3, 0, 9, 2],
        'Price': [2.55, 3.39, None, 5.95, 1.85]
    })

    print("Orijinal Veri:")
    print(data)

    # Eksik verileri temizleme
    data = handle_missing_values(data, method='fill', fill_value=0)

    # Veri türlerini düzenleme
    data = unify_column_types(data, 'InvoiceNo', target_type=str)

    print("\nÖn İşlenmiş Veri:")
    print(data)

    # LSTM ile anomali tespiti yapma
    anomalies = detect_anomalies_with_lstm(data, sequence_length=3)

if __name__ == "__main__":
    main()
