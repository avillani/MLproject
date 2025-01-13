import pandas as pd

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

    # Esegui una modifica (ad esempio, rimuovi una colonna)
    if 'State' in data.columns:
        data.drop(columns=['State'], inplace=True)
    print("Dati modificati:\n", data.head())

    # Salva il file modificato
    output_path = '../dataset/sistemato.csv' # Cambia nome se necessario
    print(f"Salvataggio nel file: {output_path}")
    data.to_csv(output_path, index=False)
    print("File salvato correttamente!")

    # Leggi di nuovo il file per confermare
    reloaded_data = pd.read_csv(output_path)
    print("Dati ricaricati:\n", reloaded_data.head())

except FileNotFoundError:
    print("Errore: il file non è stato trovato")
except pd.errors.ParserError:
    print("Errore: controlla il formato del file")
except Exception as e:
    print("Errore: ", e)