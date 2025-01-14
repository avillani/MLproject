import pandas as pd
import numpy as np

file_path = '../dataset/sistemato.csv'

try:
    # Carica il dataset
    data = pd.read_csv(file_path, low_memory=False)

    # Esamina le prime righe
    print(data.head())

    # Mostra informazioni generali sul dataset
    print(data.info())

    # Controlla la presenza di valori nulli
    print(data.isnull().sum())

    # Rimuovi una colonna)
    if 'State' in data.columns:
        data.drop(columns=['State'], inplace=True)
    print("Dati modificati:\n", data.head())

    # Filtra solo i dati relativi all'Europa
    data_europe = data[data['Region'] == 'Europe']

    # Visualizza le prime righe del nuovo dataset
    print(data_europe.head())

    # Sostituisci -99 con NaN
    data.replace(-99, np.nan, inplace=True)

    # Salva il file modificato
    output_path = '../dataset/sistemato.csv'
    print(f"Salvataggio nel file: {output_path}")
    data.to_csv(output_path, index=False)
    print("File salvato correttamente!")

    # Controlla i valori nulli
    print(data.isnull().sum())

except FileNotFoundError:
    print("Errore: il file non è stato trovato")
except pd.errors.ParserError:
    print("Errore: controlla il formato del file")
except Exception as e:
    print("Errore: ", e)