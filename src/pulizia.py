import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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

    # Conta il numero di istanze per ogni città
    city_counts = data['City'].value_counts()

    # Soglia massima di istanze per città
    MAX_INSTANCES = 9000

    # Ordinare il dataset per città e data (senza creare una colonna data)
    data = data.sort_values(by=["City", "Year", "Month", "Day"], ascending=[True, False, False, False])

    # Sottocampionare: per ogni città, teniamo solo le ultime MAX_INSTANCES righe
    data = data.groupby("City").head(MAX_INSTANCES)

    # Stampiamo la nuova distribuzione
    print(data["City"].value_counts())
    print(data["City"].value_counts().describe())

    # Seleziona solo le città con almeno 4000 istanze
    cities_to_keep = city_counts[city_counts >= 4000].index

    # Filtra il dataset mantenendo solo le città con almeno 4000 istanze
    data = data[data['City'].isin(cities_to_keep)]

    # Verifica il risultato
    print("Città rimanenti:", data['City'].unique())
    print("Dimensione del dataset dopo il filtro:", data.shape)
    print(data["City"].value_counts())
    print(data["City"].value_counts().describe())

    # Salva il file modificato
    print(f"Salvataggio nel file: {file_path}")
    data.to_csv(file_path, index=False)
    print("File salvato correttamente!")

    new_city_counts = data["City"].value_counts()

    # Visualizza la distribuzione
    plt.figure(figsize=(12, 8))
    new_city_counts.plot(kind='bar')
    plt.xlabel("Città")
    plt.ylabel("Numero di istanze")
    plt.title("Distribuzione delle istanze per città")
    plt.xticks(rotation=90)
    plt.show()

    # Stampa città con pochi dati
    threshold = 2000  # Numero minimo di istanze per città (puoi modificarlo)
    cities_low_data = city_counts[city_counts < threshold]
    print(f"Città con meno di {threshold} istanze:\n", cities_low_data)

    print(data.head())

    if 'Year' in data.columns:
        # Verifica i valori unici presenti nella colonna 'Year'
        print("Valori unici nella colonna 'Year':")
        print(data['Year'].unique())

        # Conta le occorrenze per ogni anno
        print("\nDistribuzione degli anni:")
        print(data['Year'].value_counts().sort_index())  # Ordinato per anno

        # Verifica se ci sono valori fuori dall'intervallo previsto (ad esempio, anni molto lontani)
        print("\nStatistica descrittiva sulla colonna 'Year':")
        print(data['Year'].describe())
    else:
        print("Colonna 'Year' non trovata nel dataset.")

    mean_temp = data['AvgTemperature'].mean()
    std_temp = data['AvgTemperature'].std()

    print("Media:", mean_temp)
    print("Deviazione Standard:", std_temp)


except FileNotFoundError:
    print("Errore: il file non è stato trovato")
except pd.errors.ParserError:
    print("Errore: controlla il formato del file")
except Exception as e:
    print("Errore: ", e)