import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
y_pred_rf = rf_model.predict(X_test)

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

# Salviamo il modello
joblib.dump(xgb_model, "models/xgboost_model.pkl")