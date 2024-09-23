import pandas as pd # type: ignore
import joblib # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore


train_data = pd.read_csv('C:/Users/nejat/AIM Projects/week4 data/train.csv', low_memory=False)

train_data.fillna(train_data.mean(), inplace=True)
train_data['Column7'] = train_data['Column7'].astype('category').cat.codes  

X = train_data.drop(['Sales', 'Date'], axis=1)  
y = train_data['Sales']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

model_path = 'C:/Users/nejat/AIM Projects/Rossmann_Sales_Forecast/models/random_forest_model.pkl'
joblib.dump(model, model_path)

print("Model saved successfully.")
