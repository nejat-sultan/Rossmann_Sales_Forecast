from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error # type: ignore
import joblib # type: ignore
from logs.logger import get_logger

logger = get_logger('C:/Users/nejat/AIM Projects/Rossmann_Sales_Forecast/logs/logger.log')

def train_model(train_data):
    try:
        X = train_data.drop(['Sales', 'Date'], axis=1)  
        y = train_data['Sales']

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully.")

        y_pred = model.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred)
        mae = mean_absolute_error(y_valid, y_pred)
        logger.info(f"Model Evaluation - MSE: {mse}, MAE: {mae}")

        model_path = 'C:/Users/nejat/AIM Projects/Rossmann_Sales_Forecast/models/random_forest_model.pkl'
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        return model
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise
