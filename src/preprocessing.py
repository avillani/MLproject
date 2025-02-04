import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


file_path = '../dataset/sistemato.csv'

data = pd.read_csv(file_path, low_memory=False)

label_encoder = LabelEncoder()

# Applichiamo il Label Encoding alla colonna 'City'
data['City_encoded'] = label_encoder.fit_transform(data['City'])

# Ora 'City_encoded' contiene i numeri corrispondenti alle città
print(data[['City', 'City_encoded']].head())

# Creiamo l'oggetto per la standardizzazione
scaler = StandardScaler()

# Applichiamo la standardizzazione alle colonne numeriche
data[['AvgTemperature', 'Year']] = scaler.fit_transform(data[['AvgTemperature', 'Year']])

# Verifica i primi valori per vedere come sono stati trasformati
print(data[['AvgTemperature', 'Year']].head())

# Encoding ciclico per il mese
#data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
#data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

# Encoding ciclico per il giorno
#data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31)
#data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31)

# Elimina le colonne mese e giorno originali
#data.drop(columns=['Month', 'Day'], inplace=True)

data.drop(columns=["Country"], inplace=True)


# Salva il file modificato
print(f"Salvataggio nel file: {file_path}")
data.to_csv(file_path, index=False)
print("File salvato correttamente!")