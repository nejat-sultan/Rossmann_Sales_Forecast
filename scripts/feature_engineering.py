from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from logs.logger import get_logger  

logger = get_logger('C:/Users/nejat/AIM Projects/Rossmann_Sales_Forecast/logs/logger.log')

def feature_engineering(df):
    try:
        # Example feature: Sales per Customer
        if 'Sales' in df.columns and 'Customers' in df.columns:
            df['SalesPerCustomer'] = df['Sales'] / df['Customers']
        else:
            logger.warning("Columns 'Sales' or 'Customers' are missing; skipping SalesPerCustomer calculation.")

        # Example feature: Store Type Encoding
        if 'StoreType' in df.columns:
            le = LabelEncoder()
            df['StoreTypeEncoded'] = le.fit_transform(df['StoreType'])
        else:
            logger.warning("Column 'StoreType' is missing; skipping encoding.")

        # Scaling numerical features (as an example)
        numerical_features = ['Sales', 'Customers', 'CompetitionDistance', 'Promo2SinceYear']
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features].fillna(0))

        # Additional features could be added here
        logger.info("Feature engineering completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise

