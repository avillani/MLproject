import pickle # Per salvare e caricare il LabelEncoder

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


file_path = '../dataset/sistemato.csv'

data = pd.read_csv(file_path)

# Se esiste un file con il LabelEncoder salvato, lo carichiamo
encoder_path = "../dataset/label_encoder.pkl"

print(f"Città uniche prima dell'encoding: {data['City'].nunique()}")


try:
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    print("LabelEncoder caricato!")
except FileNotFoundError:
    # Se non esiste, creiamo un nuovo LabelEncoder e lo salviamo
    label_encoder = LabelEncoder()
    label_encoder.fit(data["City"])  # Addestriamo il LabelEncoder
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print("LabelEncoder creato e salvato!")

data['City_encoded'] = label_encoder.transform(data['City'])

print(f"Città uniche dopo l'encoding: {data['City_encoded'].nunique()}")
print("Classi del LabelEncoder:", label_encoder.classes_)

missing_cities = set(data["City"].unique()) - set(label_encoder.classes_)
print(f"Città nel dataset ma non nel LabelEncoder: {missing_cities}")

# Creiamo l'oggetto per la standardizzazione
scaler = StandardScaler()

scaler.fit(data[['AvgTemperature', 'Year']])

# Applichiamo la standardizzazione alle colonne numeriche
data[['AvgTemperature', 'Year']] = scaler.transform(data[['AvgTemperature', 'Year']])

scaler_path = '../dataset/scaler.pkl'

try:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("StandardScaler caricato!")
except FileNotFoundError:
    # Se non esiste, creiamo un nuovo StandardScaler, lo fittiamo e lo salviamo
    scaler = StandardScaler()
    scaler.fit(data[['AvgTemperature', 'Year']])  # Fittiamo lo scaler solo sulla colonna Year
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print("StandardScaler creato e salvato!")

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

data["Season"] = data["Month"].apply(assign_season)

# Encoding ciclico per il mese
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

# Encoding ciclico per il giorno
data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31)
data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31)

# Elimina le colonne originali
data.drop(columns=['Month', 'Day'], inplace=True)

data.drop(columns=["Country"], inplace=True)

print(f"Città uniche dopo la standardizzazione: {data['City_encoded'].nunique()}")
print(f"Righe con NaN dopo l'encoding: {data.isna().sum().sum()}")
print(f"Città uniche nel dataset finale: {data['City_encoded'].nunique()}")


data.drop(columns=["City"], inplace=True)

print(data.dtypes)

# Verifica la forma del dataframe (numero di righe e colonne)
print(f"Numero di righe: {data.shape[0]}, Numero di colonne: {data.shape[1]}")

# Verifica che non ci siano righe con valori NaN in colonne critiche
print(f"Righe con NaN prima del salvataggio: {data.isna().sum().sum()}")

import os

try:
    # Verifica se il percorso del file esiste
    if not os.path.exists(os.path.dirname(file_path)):
        raise FileNotFoundError(f"La directory {os.path.dirname(file_path)} non esiste.")

    # Salva il dataset nel file
    data.to_csv(file_path, index=False, mode='w', sep=',', encoding='utf-8')
    print("File salvato correttamente!")

except FileNotFoundError as fnf_error:
    print(f"Errore: La directory del file non esiste. Dettagli: {fnf_error}")

except PermissionError as perm_error:
    print(f"Errore di permesso: Non puoi scrivere il file in questa posizione. Dettagli: {perm_error}")

except MemoryError as mem_error:
    print(f"Errore di memoria: Il sistema ha esaurito la memoria. Dettagli: {mem_error}")

except Exception as e:
    print(f"Si è verificato un errore durante il salvataggio del file: {e}")


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