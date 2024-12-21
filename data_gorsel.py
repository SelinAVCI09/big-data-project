import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Veri Seti Özeti Fonksiyonu
def report_data_overview(data):
    print("\n--- Veri Setinin Genel Özellikleri ---\n")
    print("1. İlk 5 Satır:")
    print(data.head())

    print("\n2. Veri Seti Bilgisi:")
    print(data.info())

    print("\n3. Eksik Değerler:")
    print(data.isnull().sum())

    print("\n4. Betimsel İstatistikler:")
    print(data.describe(include='all'))  # Tüm sütunlar için betimsel istatistikler

# Histogram Çizme Fonksiyonu
def plot_histograms(data, numeric_columns):
    print("\n--- Histogram Grafikleri ---\n")
    for column in numeric_columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], kde=True, color='skyblue', bins=30)
        plt.title(f'{column} Dağılımı (Histogram)')
        plt.xlabel(column)
        plt.ylabel('Frekans')
        plt.tight_layout()
        plt.show()

# Korelasyon Haritası (Heatmap)
def plot_heatmap(data, numeric_columns):
    print("\n--- Korelasyon Haritası (Heatmap) ---\n")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Değişkenler Arası Korelasyon")
    plt.show()

# Scatter Plot Çizme Fonksiyonu
def plot_scatter(data, x_column, y_column):
    print(f"\n--- Scatter Plot: {x_column} vs {y_column} ---\n")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_column, y=y_column, data=data, color='purple', alpha=0.7)
    plt.title(f"{x_column} ile {y_column} Arasındaki İlişki")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()
    plt.show()

# Outlier Tespiti
def detect_and_handle_outliers(data, numeric_columns):
    print("\n--- Outlier Tespiti ve İşlenmesi ---\n")
    for column in numeric_columns:
        # IQR (Interquartile Range) yöntemi ile aykırı değerleri tespit etme
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        print(f"{column} için tespit edilen aykırı değerler:")
        print(outliers)
        
        # Outlier'ları silme (isteğe bağlı)
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        
    return data

# Ana Fonksiyon
def main():
    # Veri seti yolu
    data_path = os.path.join("data", "OnlineRetail.csv")

    # OnlineRetail.csv dosyasını okuma
    try:
        data = pd.read_csv(data_path, encoding='latin1')
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {data_path}. Lütfen doğru yolu belirtin.")
        return

    # Veri setinin genel görünümü
    report_data_overview(data)

    # Eksik verileri temizleme
    print("\n--- Eksik Değerler Temizleniyor... ---")
    data_cleaned = data.fillna(0)  # Eksik değerler 0 ile dolduruluyor

    print("\n--- Temizlenmiş Veri Seti ---")
    print(data_cleaned)  # Temizlenmiş veriyi yazdırıyoruz

    # Veri türlerini düzenleme
    data_cleaned['InvoiceNo'] = data_cleaned['InvoiceNo'].astype(str)
    data_cleaned['CustomerID'] = data_cleaned['CustomerID'].astype(str)

    print("\n--- Temizlenmiş Veri Seti: Veri Türleri Düzenlenmiş ---")
    print(data_cleaned.head())

    # Sayısal sütunları belirleme
    numeric_columns = ['Quantity', 'UnitPrice']

    # Histogram Grafikleri
    plot_histograms(data_cleaned, numeric_columns)

    # Korelasyon Haritası
    plot_heatmap(data_cleaned, numeric_columns)

    # Scatter Plot: Quantity vs UnitPrice
    plot_scatter(data_cleaned, 'Quantity', 'UnitPrice')

    # Outlier Tespiti ve İşlenmesi
    data_cleaned = detect_and_handle_outliers(data_cleaned, numeric_columns)

    # Verinin normalize edilmesi veya standartlaştırılması
    print("\n--- Verinin Normalize Edilmesi veya Standartlaştırılması ---")
    scaler = MinMaxScaler()
    data_cleaned['UnitPrice'] = scaler.fit_transform(data_cleaned[['UnitPrice']])
    scaler = StandardScaler()
    data_cleaned['Quantity'] = scaler.fit_transform(data_cleaned[['Quantity']])

    # Kateğorik Değişkenlerin Sayısal Hale Getirilmesi (Encoding)
    print("\n--- Kateğorik Değişkenlerin Sayısal Hale Getirilmesi (Encoding) ---")
    label_encoder = LabelEncoder()
    data_cleaned['Country'] = label_encoder.fit_transform(data_cleaned['Country'])

    print("\n--- Temizlenmiş ve İşlenmiş Veri Seti ---")
    print(data_cleaned.head())

if __name__ == "__main__":
    main()
