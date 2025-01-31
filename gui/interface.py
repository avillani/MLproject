import streamlit as st
from datetime import date


def main():
    st.set_page_config(page_title="MeteoMind", page_icon="🌡️", layout="centered")

    # Caricamento logo
    import os
    st.image(os.path.join(os.path.dirname(__file__), "logo.png"), width=150)

    # Titolo
    st.title("Previsione della Temperatura")

    # Selezione della città
    city = st.selectbox("Seleziona una città:", ["Londra", "Parigi", "Roma", "Madrid", "Berlino"])

    # Selezione della data
    forecast_date = st.date_input("Seleziona una data futura:", min_value=date.today())

    # Bottone per generare la previsione
    if st.button("Prevedi Temperatura"):
        # Qui andrà il codice per la previsione del modello
        st.success(f"La temperatura prevista per {city} il {forecast_date} è di XX°C")


if __name__ == "__main__":
    main()