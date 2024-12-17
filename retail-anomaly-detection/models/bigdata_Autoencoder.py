# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:21:06 2024

@author: pc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

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
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()
    plt.show()

# Autoencoder ile Anomali Tespiti
def train_autoencoder(data):
    # Veriyi ölçekleme
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Eğitim ve test veri setlerini ayırma
    X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

    # Modeli oluşturma
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(X_train.shape[1])
    ])

    model.compile(optimizer='adam', loss='mse')

    # Modeli eğitme
    model.fit(X_train, X_train, epochs=50, batch_size=16, verbose=1)

    # Yeniden yapılandırma hatalarını hesaplama
    X_test_pred = model.predict(X_test, verbose=0)
    reconstruction_error = np.mean(np.abs(X_test - X_test_pred), axis=1)

    # Hata dağılımını görselleştirme ve eşik belirleme
    threshold = np.percentile(reconstruction_error, 95)
    plt.hist(reconstruction_error, bins=50)
    plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    plt.legend()
    plt.title("Reconstruction Error Distribution")
    plt.show()

    # Anomalileri işaretleme
    anomalies = reconstruction_error > threshold
    print(f"Total anomalies detected: {np.sum(anomalies)}")
    if np.sum(anomalies) == 0:
        print("No anomalies detected. All data points are within the threshold.")
    print(f"Anomalies: {anomalies}")

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
    data['Price'] = data['Price'].fillna(data['Price'].mean())
    data = handle_missing_values(data, method='fill', fill_value=0)

    # Veri türlerini düzenleme
    data = unify_column_types(data, 'InvoiceNo', target_type=str)

    print("\nÖn İşlenmiş Veri:")
    print(data)

    # Grafikler
    print("\nGörselleştirme: StockCode'e göre Quantity (Bar Grafiği)")
    visualize_data(data, x_column='StockCode', y_column='Quantity', plot_type='bar')

    print("\nGörselleştirme: StockCode'e göre Price (Çizgi Grafiği)")
    visualize_data(data, x_column='StockCode', y_column='Price', plot_type='line')

    print("\nGörselleştirme: Quantity'e göre Price (Scatter Grafiği)")
    visualize_data(data, x_column='Quantity', y_column='Price', plot_type='scatter')

    # Anomali tespiti
    numeric_data = data[['Quantity', 'Price']]
    train_autoencoder(numeric_data)

if __name__ == "__main__":
    main()
