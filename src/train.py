import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
