import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
import os

# Veriyi yükle
data_path = os.path.join("data", "OnlineRetail.csv")

# Veri setinin encode hatasını kontrol et
df = pd.read_csv(data_path, encoding='ISO-8859-1')

# Gerekli sütunları seç
df = df[['Quantity', 'UnitPrice', 'CustomerID']].dropna()

# Veriyi standardize et
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Quantity', 'UnitPrice']])  # Sadece sayısal kolonları kullan

# Anomali etiketleri için threshold belirleme
threshold = 3  # Z-score için belirli bir eşik
outliers = np.where(scaled_data > threshold, 1, 0)

# Eğitim ve test veri setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(scaled_data, outliers, test_size=0.2, random_state=42)

# 1. Random Forest Anomali Tespiti Modeli
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = rf.predict(X_test)

# Random Forest sonuçlarını yazdır
print("Random Forest Modeli Sonuçları:")
print(classification_report(y_test, y_pred))

# 2. Autoencoder Anomali Tespiti Modeli

# Autoencoder modeli oluştur
autoencoder = Sequential()
autoencoder.add(Dense(64, activation='relu', input_shape=(scaled_data.shape[1],)))
autoencoder.add(Dense(32, activation='relu'))
autoencoder.add(Dense(64, activation='relu'))
autoencoder.add(Dense(scaled_data.shape[1], activation='sigmoid'))

# Modeli derle
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğit
autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, validation_data=(X_test, X_test))

# Aykırı değerleri tespit et
reconstructed = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)
threshold_autoencoder = np.percentile(mse, 95)  # %95'lik dilimdeki hata eşik değeri
outliers_autoencoder = mse > threshold_autoencoder

# Autoencoder sonuçlarını yazdır
print("Autoencoder Modeli Sonuçları:")
print(f'Autoencoder ile tespit edilen anomali oranı: {np.mean(outliers_autoencoder)}')

# Anomali tespiti görselleştirmesi
plt.figure(figsize=(12,6))

# Random Forest görselleştirmesi
plt.subplot(1, 2, 1)
plt.title('Random Forest Aykırı Değer Tahminleri')
plt.scatter(range(len(y_test)), y_test, c=y_pred, cmap='coolwarm')
plt.xlabel('Test Veri Örnekleri')
plt.ylabel('Gerçek Aykırı Değer Etiketleri')

# Autoencoder görselleştirmesi
plt.subplot(1, 2, 2)
plt.title('Autoencoder Aykırı Değer Tahminleri')
plt.scatter(range(len(mse)), mse, c=outliers_autoencoder, cmap='coolwarm')
plt.xlabel('Test Veri Örnekleri')
plt.ylabel('Rekonstrüksiyon Hatası (MSE)')

plt.tight_layout()
plt.show()
 