# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:57:07 2024

@author: pc
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        sns.barplot(x=x_column, y=y_column, data=data, ci=None)
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

    # Grafikler
    print("\nGörselleştirme: StockCode'e göre Quantity (Bar Grafiği)")
    visualize_data(data, x_column='StockCode', y_column='Quantity', plot_type='bar')

    print("\nGörselleştirme: StockCode'e göre Price (Çizgi Grafiği)")
    visualize_data(data, x_column='StockCode', y_column='Price', plot_type='line')

    print("\nGörselleştirme: Quantity'e göre Price (Scatter Grafiği)")
    visualize_data(data, x_column='Quantity', y_column='Price', plot_type='scatter')

if __name__ == "__main__":
    main()
