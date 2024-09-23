import joblib # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
import numpy as np

model = RandomForestRegressor()
X = np.random.rand(100, 5)
y = np.random.rand(100)
model.fit(X, y)

joblib.dump(model, 'C:/Users/nejat/AIM Projects/Rossmann_Sales_Forecast/models/random_forest_model.pkl')