# Rossmann Sales Forecast

## Overview

This project explores customer purchasing behavior for the Rossmann store sales dataset. The goal is to perform exploratory data analysis (EDA) to understand how promotions, store openings, and other factors influence sales, as well as to predict future sales.

## Introduction

The purpose of this project is to perform EDA on the Rossmann store sales dataset to answer questions related to customer behavior and sales performance. Key areas of exploration include:

- Distribution of promotions between training and test sets
- Sales behavior around holidays
- Seasonal purchasing trends
- Correlation between sales and customer count
- Impact of promotions on sales
- Store opening and closing times

## Data Sources

The data for this project can be obtained from the [Kaggle Rossmann Store Sales Competition](https://www.kaggle.com/competitions/rossmann-store-sales/data). The following files are used:

- `train.csv`: Historical sales data
- `test.csv`: Test data for prediction
- `store.csv`: Store-related information

## Installation

1. Clone the repository:
   git clone https://github.com/nejat-sultan/Rossmann_Sales_Forecast.git
   cd Rossmann_Sales_Forecast
   pip install -r requirements.txt

## Data Cleaning and Processing

The `data_cleaning.py` script handles the following tasks:

- Loading the datasets
- Cleaning missing values
- Creating additional features (e.g., date features)
- Merging training and test datasets with store information
- Logging steps using the logger library for traceability

## Task 1 - Exploratory Data Analysis (EDA)

EDA will be conducted in a separate Jupyter Notebook. Key analysis areas will include:

- Visualizing the impact of promotions on sales
- Examining seasonal trends
- Analyzing customer behavior relative to store operations

## Task 2 - Prediction of Store Sales

Prediction of sales is the central task in this challenge. The objective is to predict daily sales in various stores up to 6 weeks ahead of time, enabling the company to plan ahead.

### 2.1 Preprocessing

- Convert all non-numeric columns to numeric, handle NaN values, and generate new features from existing features.
- Extract features from datetime columns:
  - Weekdays and weekends
  - Number of days to and after holidays
  - Beginning, mid-month, and end of the month
  - Additional relevant features
- Scale the data using the standard scaler from sklearn.

### 2.2 Building Models with Sklearn Pipelines

- Utilize tree-based algorithms, such as Random Forest Regressor, to model the data.
- Implement sklearn pipelines for modular and reproducible modeling.

### 2.3 Choose a Loss Function

- Choose and defend an appropriate loss function for the sales prediction task.

### 2.4 Post Prediction Analysis

- Explore feature importance and estimate the confidence interval of predictions creatively.

### 2.5 Serialize Models

- Save trained models with timestamps (e.g., `10-08-2020-16-32-31-00.pkl`) for tracking predictions.

### 2.6 Building Model with Deep Learning

- Create a Long Short-Term Memory (LSTM) model using TensorFlow or PyTorch.
- Transform the dataset into time series data and perform necessary preprocessing steps.
- Build the LSTM Regression model to predict future sales.

## Task 3 - Model Serving API Call

Create a REST API to serve the trained machine-learning models for real-time predictions.

- **Choose a Framework**: Select a suitable framework for building REST APIs (e.g., Flask, FastAPI).
- **Load the Model**: Use the serialized model from Task 2 to load the trained model.
- **Define API Endpoints**: Create endpoints to accept input data and return predictions.
- **Handle Requests**: Implement logic to preprocess input data and make predictions using the loaded model.
- **Return Predictions**: Format and return predictions as a response to API calls.
- **Deployment**: Deploy the API to a web server or cloud platform.

## Logging

The project uses the logger library to record important steps and errors during execution. Logs can be found in the `logs` directory.
