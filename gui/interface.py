import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import os
import joblib  # Per caricare il modello

# Funzione per caricare il modello
def load_model():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Percorso dello script corrente
        file_path = os.path.join(script_dir, "..", "src", "models", "ensemble_model.pkl")
        model = joblib.load(file_path)
        print("Modello caricato correttamente!")
        print(f"Tipo di modello caricato: {type(model)}")
        print(f"Contenuto del modello: {model}")
        return model
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        return None


# Codice per fare la previsione combinata
def predict_with_ensemble(model_tuple, features,  scaler):
    model_xgb, model_rf, weight_xgb, weight_rf = model_tuple

    # Verifica che i modelli siano fittati
    if hasattr(model_xgb, "feature_importances_") and hasattr(model_rf, "feature_importances_"):
        print("Modelli correttamente fittati!")
    else:
        print("I modelli non sono fittati.")

    # Previsione di XGBoost
    pred_xgb = model_xgb.predict([features])

    # Previsione di RandomForest
    pred_rf = model_rf.predict([features])

    # Combinazione delle previsioni pesate
    final_prediction = (weight_xgb * pred_xgb) + (weight_rf * pred_rf)

    print("Predizione finale:", final_prediction)

    # Denormalizziamo la temperatura predetta
    temp_original = (final_prediction[0] * 8.41809632218773) + 11.092233505521685

    print(f"Temperatura denormalizzata: {temp_original}")
    
    return temp_original

# Funzione per caricare le città dal dataset e creare la mappatura
def load_city_mapping():
    # Carica il dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Percorso dello script corrente
    file_path = os.path.join(script_dir, "..", "dataset", "city.csv")
    data = pd.read_csv(file_path)

    # Creiamo il dizionario di mappatura tra codice città e nome
    city_map = dict(zip(data['City_encoded'], data['City']))

    return city_map


# Funzione per preprocessare la data
def preprocess_data(city_encoded, forecast_date, scaler):
    # Normalizzazione dell'anno
    year = np.array([[forecast_date.year]])  # Lo trasformiamo in array per compatibilità con lo scaler
    year_normalized = scaler.transform(np.array([[0, forecast_date.year]]))[0, 1]  # Usiamo solo la colonna "Year"

    # Calcolo della stagione
    month = forecast_date.month
    if month in [12, 1, 2]:
        season = 0  # Inverno
    elif month in [3, 4, 5]:
        season = 1  # Primavera
    elif month in [6, 7, 8]:
        season = 2  # Estate
    else:
        season = 3  # Autunno

    converted = 1

    # Encoding ciclico per il mese
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Encoding ciclico per il giorno
    day_sin = np.sin(2 * np.pi * forecast_date.day / 31)
    day_cos = np.cos(2 * np.pi * forecast_date.day / 31)

    # Restituiamo le caratteristiche preprocessate
    return [year_normalized, converted, city_encoded, season, month_sin, month_cos, day_sin, day_cos]

def load_scaler():
    scaler_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "scaler.pkl")
    with open(scaler_path, "rb") as f:
        return joblib.load(scaler_path)

def main():
    st.set_page_config(page_title="MeteoMind", page_icon="🌡️", layout="centered")

    # Caricamento logo
    import os
    st.image(os.path.join(os.path.dirname(__file__), "logo.png"), width=150)

    # Titolo
    st.title("Previsione della Temperatura")

    # Carica la mappatura delle città
    city_map = load_city_mapping()

    # Selezione della città con i codici numerici
    city_encoded = st.selectbox("Seleziona una città:", list(city_map.keys()), format_func=lambda x: city_map[x])

    # Selezione della data
    forecast_date = st.date_input("Seleziona una data futura:", min_value=date.today())

    # Carichiamo lo scaler
    scaler = load_scaler()

    # Bottone per generare la previsione
    if st.button("Prevedi Temperatura"):
        model_tuple = load_model()  # Carica il modello solo quando serve
        features = preprocess_data(city_encoded, forecast_date, scaler)
        predicted_temp = predict_with_ensemble(model_tuple, features, scaler)
        st.success(
            f"La temperatura prevista per {city_map[city_encoded]} il {forecast_date} è di {predicted_temp:.2f}°C")


if __name__ == "__main__":
    main()