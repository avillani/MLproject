import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = '../dataset/city.csv'

# Carica il dataset
data = pd.read_csv(file_path, low_memory=False)

# Rimuovi una colonna
if 'State' in data.columns:
    data.drop(columns=['State'], inplace=True)

# Controlla la presenza della colonna 'Region' prima di filtrare
if 'Region' in data.columns:
    data = data[data['Region'] == 'Europe']
    data.drop(columns=['Region'], inplace=True)

# Sostituisci -99 con NaN
data.replace(-99, np.nan, inplace=True)

# Rimuove tutte le tuple dove City ha valore Hamburg
data = data[data['City'] != 'Hamburg']

# Rimuove solo le tuple con valori nulli nelle altre città
data = data.dropna(subset=['AvgTemperature'])

# Conta il numero di istanze per ogni città
city_counts = data['City'].value_counts()

print(data["City"].value_counts())

# Seleziona solo le città con almeno 4000 istanze
cities_to_keep = city_counts[city_counts >= 4000].index

# Filtra il dataset mantenendo solo le città con almeno 4000 istanze
data = data[data['City'].isin(cities_to_keep)]

print(data["City"].value_counts())

# Mantieni solo la colonna 'City'
data = data[['City']].drop_duplicates()

label_encoder = LabelEncoder()

# Applichiamo il Label Encoding alla colonna 'City'
data['City_encoded'] = label_encoder.fit_transform(data['City'])

print(data["City"].value_counts())


data.to_csv(file_path, index=False)