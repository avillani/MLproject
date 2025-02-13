import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import xgboost as xgb

# Caricare i dati preprocessati
train_data = pd.read_csv("../dataset/train.csv")
test_data = pd.read_csv("../dataset/test.csv")

# Caricare direttamente le feature e il target dal preprocessing
X_train = train_data.drop(columns=["AvgTemperature"])
y_train = train_data["AvgTemperature"]
X_test = test_data.drop(columns=["AvgTemperature"])
y_test = test_data["AvgTemperature"]

# Creare e addestrare il modello
model = LinearRegression()
model.fit(X_train, y_train)

# Salvare il modello addestrato
joblib.dump(model, "models/linear_regression.pkl")

# Fare previsioni sul test set
y_pred = model.predict(X_test)

# Calcolare le metriche di valutazione
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Stampare i risultati
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


# Creare il modello Random Forest
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)

# Addestrare il modello
rf_model.fit(X_train, y_train)

# Fare previsioni sul test set
y_pred_rf = rf_model.predict(X_test)

# Calcolare le metriche di valutazione
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# Stampare i risultati
print(f"Random Forest - Mean Absolute Error (MAE): {mae_rf:.4f}")
print(f"Random Forest - Root Mean Squared Error (RMSE): {rmse_rf:.4f}")
print(f"Random Forest - R² Score: {r2_rf:.4f}")


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Addestrare il modello
rf_model.fit(X_train, y_train)

# Fare previsioni sul test set
rf_pred = rf_model.predict(X_test)

# Calcolare le metriche di valutazione
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# Stampare i risultati
print(f"Random Forest - Mean Absolute Error (MAE): {mae_rf:.4f}")
print(f"Random Forest - Root Mean Squared Error (RMSE): {rmse_rf:.4f}")
print(f"Random Forest - R² Score: {r2_rf:.4f}")



# Creiamo il modello XGBoost
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,  # Numero di alberi
    learning_rate=0.1,  # Tasso di apprendimento
    max_depth=6,  # Profondità degli alberi
    subsample=0.8,  # Percentuale di dati usati per ogni albero
    colsample_bytree=0.8  # Percentuale di feature usate per ogni albero
)

# Alleniamo il modello
xgb_model.fit(X_train, y_train)

# Facciamo le predizioni
y_pred_xgb = xgb_model.predict(X_test)

# Calcoliamo le metriche
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

# Stampiamo i risultati
print(f'XGBoost - Mean Absolute Error (MAE): {mae_xgb:.4f}')
print(f'XGBoost - Root Mean Squared Error (RMSE): {rmse_xgb:.4f}')
print(f'XGBoost - R² Score: {r2_xgb:.4f}')


# Definiamo gli iperparametri da testare
param_dist = {
    'n_estimators': [50, 100, 200, 300],  # Numero di alberi
    'max_depth': [3, 5, 7, 10],  # Profondità massima degli alberi
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Tasso di apprendimento
    'subsample': [0.6, 0.8, 1.0],  # Percentuale di dati usati in ogni iterazione
    'colsample_bytree': [0.6, 0.8, 1.0]  # Percentuale di feature usate per ogni albero
}

# Inizializziamo il modello
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Randomized Search con validazione incrociata
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,  # Numero di combinazioni casuali da provare
    cv=3,  # Validazione incrociata a 3 fold
    scoring='neg_mean_squared_error',  # Ottimizziamo per MSE
    n_jobs=-1,  # Usa tutti i core disponibili
    verbose=2
)

# Eseguiamo la ricerca
random_search.fit(X_train, y_train)

# Migliori iperparametri trovati
best_params = random_search.best_params_
print("Migliori iperparametri trovati:", best_params)

# Alleniamo il modello con i migliori iperparametri
best_xgb = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
best_xgb.fit(X_train, y_train)

# Valutiamo il modello
y_pred_xgb = best_xgb.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'XGBoost Ottimizzato - MAE: {mae_xgb:.4f}')
print(f'XGBoost Ottimizzato - RMSE: {rmse_xgb:.4f}')
print(f'XGBoost Ottimizzato - R² Score: {r2_xgb:.4f}')

# Parametri da testare per il modello XGBoost
param_dist = {
    'n_estimators': np.arange(100, 500, 100),  # Numero di alberi
    'learning_rate': np.linspace(0.01, 0.2, 5),  # Tasso di apprendimento
    'max_depth': np.arange(3, 10, 1),  # Profondità massima degli alberi
    'subsample': np.linspace(0.6, 1, 5),  # Percentuale di dati usati in ogni iterazione
    'colsample_bytree': np.linspace(0.6, 1, 5),  # Percentuale di feature per albero
    'gamma': np.linspace(0, 2, 5),  # Aggiunto gamma per controllare la complessità
    'reg_alpha': np.logspace(-3, 1, 5),  # Aggiunto reg_alpha per la regolarizzazione
    'reg_lambda': np.logspace(-3, 1, 5)  # Aggiunto reg_lambda per la regolarizzazione
}

# Inizializziamo il modello base XGBoost
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Randomized Search con validazione incrociata
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=30,  # Numero di combinazioni da provare (ridotto)
    scoring='neg_mean_absolute_error',  # Ottimizziamo per MAE
    cv=3,  # Validazione incrociata a 3 fold
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Eseguiamo la ricerca
random_search.fit(X_train, y_train)

# Migliori iperparametri trovati
best_params = random_search.best_params_
print("Migliori iperparametri trovati:", best_params)

# Alleniamo il modello con i migliori iperparametri
best_xgb = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
best_xgb.fit(X_train, y_train)

# Valutiamo il modello
xgb_pred = best_xgb.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'XGBoost Ottimizzato - MAE: {mae_xgb:.4f}')
print(f'XGBoost Ottimizzato - RMSE: {rmse_xgb:.4f}')
print(f'XGBoost Ottimizzato - R² Score: {r2_xgb:.4f}')

# Combinazione pesata
ensemble_predictions = 0.60 * y_pred_xgb + 0.40 * rf_pred

# Calcolare il MAE per l'ensemble
ensemble_mae = mean_absolute_error(y_test, ensemble_predictions)

# Calcolare il RMSE per l'ensemble
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions))

print(f"Ensemble MAE: {ensemble_mae}")
print(f"Ensemble RMSE: {ensemble_rmse}")
ensemble_r2 = r2_score(y_test, ensemble_predictions)
print("Ensemble R²:", ensemble_r2)

# Salviamo il modello ensemble
joblib.dump((xgb_model, rf_model, 0.6, 0.4), "models/ensemble_model.pkl", compress=7)