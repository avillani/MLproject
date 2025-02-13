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

# Label Encoding per le stagioni
def assign_season(month):
    if month in [12, 1, 2]:
        return 0  # Inverno
    elif month in [3, 4, 5]:
        return 1  # Primavera
    elif month in [6, 7, 8]:
        return 2  # Estate
    else:
        return 3  # Autunno

#data["Season"] = data["Month"].apply(assign_season)

# Encoding ciclico per il mese
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

# Encoding ciclico per il giorno
data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31)
data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31)

# Elimina le colonne mese e giorno originali
data.drop(columns=['Month', 'Day'], inplace=True)

data.drop(columns=["Country"], inplace=True)

data.drop(columns=["City"], inplace=True)

# Salva il file modificato
print(f"Salvataggio nel file: {file_path}")
data.to_csv(file_path, index=False)
print("File salvato correttamente!")

# Selezioniamo il 80% dei dati più vecchi per il training e il 20% più recenti per il test
train_size = int(0.8 * len(data))  # 80% dei dati
df_train = data[:train_size]  # Training set
df_test = data[train_size:]    # Test set

# Separa le variabili di input (X) e target (y) per il training e test set
X_train = df_train.drop(columns=['AvgTemperature'])
y_train = df_train['AvgTemperature']
X_test = df_test.drop(columns=['AvgTemperature'])
y_test = df_test['AvgTemperature']

# Salvare i dataset in formato CSV
df_train.to_csv("../dataset/train.csv", index=False)
df_test.to_csv("../dataset/test.csv", index=False)

# Verifica delle dimensioni
print(f'Training set size: {X_train.shape}')
print(f'Test set size: {X_test.shape}')

