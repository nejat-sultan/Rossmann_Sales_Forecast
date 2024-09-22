import pandas as pd  # type: ignore
from logs.logger import get_logger  # type: ignore

logger = get_logger('C:/Users/nejat/AIM Projects/Rossmann_Sales_Forecast/logs/logger.log')

def load_data(train_path, test_path, store_path):
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        store = pd.read_csv(store_path)
        logger.info("Data loaded successfully.")
        
        # Log the columns of the DataFrames
        logger.info(f"Train columns: {train.columns.tolist()}")
        logger.info(f"Test columns: {test.columns.tolist()}")
        logger.info(f"Store columns: {store.columns.tolist()}")
        
        return train, test, store
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def clean_data(df):
    try:
        # Handle missing values
        if 'CompetitionDistance' in df.columns:
            df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
        else:
            logger.warning("'CompetitionDistance' column is missing; skipping filling.")

        if 'CompetitionOpenSinceMonth' in df.columns:
            df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)

        if 'CompetitionOpenSinceYear' in df.columns:
            df['CompetitionOpenSinceYear'].fillna(0, inplace=True)

        if 'Promo2SinceWeek' in df.columns:
            df['Promo2SinceWeek'].fillna(0, inplace=True)

        if 'Promo2SinceYear' in df.columns:
            df['Promo2SinceYear'].fillna(0, inplace=True)

        if 'PromoInterval' in df.columns:
            df['PromoInterval'].fillna('None', inplace=True)

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

            # Additional Date Features
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        else:
            logger.warning("'Date' column is missing; skipping date processing.")

        logger.info("Data cleaned successfully.")
        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise


def merge_data(train, test, store):
    try:
        train = train.merge(store, on='Store', how='left')
        test = test.merge(store, on='Store', how='left')
        logger.info("Data merged successfully.")
        return train, test
    except Exception as e:
        logger.error(f"Error merging data: {e}")
        raise


def main():
    try:
        train_path = 'C:/Users/nejat/AIM Projects/week4 data/train.csv'
        test_path = 'C:/Users/nejat/AIM Projects/week4 data/test.csv'
        store_path = 'C:/Users/nejat/AIM Projects/week4 data/store.csv'
        
        train, test, store = load_data(train_path, test_path, store_path)

        train = clean_data(train)
        test = clean_data(test)

        train, test = merge_data(train, test, store)
        
        logger.info("Pipeline executed successfully.")
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")

if __name__ == "__main__":
    main()
