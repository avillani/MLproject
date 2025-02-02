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

    # Rimuovi una colonna
    if 'State' in data.columns:
        data.drop(columns=['State'], inplace=True)
    print("Dati modificati:\n", data.head())

    # Controlla la presenza della colonna 'Region' prima di filtrare
    if 'Region' in data.columns:
        data = data[data['Region'] == 'Europe']
        data.drop(columns=['Region'], inplace=True)
    else:
        print("Colonna 'Region' non trovata.")

    # Sostituisci -99 con NaN
    data.replace(-99, np.nan, inplace=True)

    # Controlla i valori nulli
    print(data.isnull().sum())


    # Conversione dati Fahrenheit in Celsius
    if 'Converted' not in data.columns:   # Controlla se la conversione è già stata fatta
        data['AvgTemperature'] = (data['AvgTemperature'] - 32) * 5 / 9
        data['Converted'] = True  # Aggiunge una colonna per segnare che è stato convertito
        print("Conversione effettuata.")
    else:
        print("I dati sono già stati convertiti in Celsius.")


    # Restituisce le tuple dove abbiamo valori nulli per la temperatura
    # e poi conta queste tuple raggruppandole sull'attributo City
    print(data[data['AvgTemperature'].isna()]['City'].value_counts())

    # Conta le istanze per ogni città PRIMA della rimozione
    city_counts_before = data["City"].value_counts()

    # Conta le istanze per ogni città che hanno un valore nullo nella temperatura
    city_null_counts = data[data["AvgTemperature"].isna()]["City"].value_counts()

    # Confronto per le città con più valori nulli
    cities_to_check = ["Hamburg", "Tirana"]
    for city in cities_to_check:
        total = city_counts_before.get(city, 0)
        missing = city_null_counts.get(city, 0)
        remaining = total - missing
        print(f"{city}: totale {total}, nulli {missing}, rimasti {remaining}")

    # Vedere la distribuzione complessiva PRIMA e DOPO (virtualmente, senza cancellare)
    print("\nDistribuzione città PRIMA della rimozione:")
    print(city_counts_before.describe())

    print("\nDistribuzione città DOPO (ipotetica) se rimuovessimo i NaN:")
    print((city_counts_before - city_null_counts).dropna().describe())

    # Rimuove tutte le tuple dove City ha valore Hamburg
    data = data[data['City'] != 'Hamburg']

    # Rimuove solo le tuple con valori nulli nelle altre città
    data = data.dropna(subset=['AvgTemperature'])

    # Conferma dei cambiamenti
    print("Dati rimanenti dopo la pulizia:")
    print(data['City'].value_counts())

    # Salva il file modificato
    output_path = '../dataset/sistemato.csv'
    print(f"Salvataggio nel file: {output_path}")
    data.to_csv(output_path, index=False)
    print("File salvato correttamente!")

    print(data.head())

except FileNotFoundError:
    print("Errore: il file non è stato trovato")
except pd.errors.ParserError:
    print("Errore: controlla il formato del file")
except Exception as e:
    print("Errore: ", e)