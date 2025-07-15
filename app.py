import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# Define the path for the model and data file
# Ensure 'random_forest_regressor_model.pkl' and 'retail_price.csv' are in the same directory as app.py
MODEL_PATH = 'random_forest_regressor_model.pkl'
DATA_PATH = 'retail_price.csv'

# Load the trained model
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure it's in the same directory.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the original dataset to get category names for one-hot encoding consistency
try:
    df_original = pd.read_csv(DATA_PATH)
    # Get unique product categories for one-hot encoding reference
    # Assuming 'product_category_name' is a column in retail_price.csv
    if 'product_category_name' in df_original.columns:
        product_categories = sorted(df_original['product_category_name'].unique().tolist())
        # Remove the first category if drop_first=True was used during training
        # This assumes the first category alphabetically was dropped
        if product_categories:
            # Recreate the dummy DataFrame to get all possible one-hot encoded columns
            temp_df_for_dummies = pd.DataFrame(columns=['product_category_name'])
            temp_df_for_dummies['product_category_name'] = product_categories
            all_dummy_cols_df = pd.get_dummies(temp_df_for_dummies, columns=['product_category_name'], drop_first=True)
            # Store the column names for later use in prediction
            expected_category_cols = all_dummy_cols_df.columns.tolist()
        else:
            expected_category_cols = []
    else:
        product_categories = []
        expected_category_cols = []
    print(f"Original data loaded successfully from {DATA_PATH} for category reference.")
    print(f"Discovered product categories: {product_categories}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please ensure it's in the same directory.")
    df_original = None
    product_categories = []
    expected_category_cols = []
except Exception as e:
    print(f"Error loading data for categories: {e}")
    df_original = None
    product_categories = []
    expected_category_cols = []

# Define the features that the model expects, in the correct order
# This list must match the order of columns in X_train during model training
# The boolean columns from one-hot encoding will be appended dynamically
# Ensure this list is consistent with the notebook's X definition
BASE_FEATURES = [
    'qty', 'freight_price', 'product_name_lenght', 'product_description_lenght',
    'product_photos_qty', 'product_weight_g', 'product_score', 'customers',
    'weekday', 'weekend', 'holiday', 'month', 'year', 's', 'volume',
    'comp_1', 'ps1', 'fp1', 'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price'
]

@app.route('/')
def home():
    """Renders the main HTML page for price prediction."""
    # Pass product_categories to the HTML template for dropdown
    return render_template('index.html', product_categories=product_categories)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests.
    Expects a JSON payload with product features.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    data = request.get_json(force=True)

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data])

    # --- Preprocessing for prediction ---
    # 1. Convert 'month_year' to datetime and extract month/year if present in input
    # For a prediction app, we might expect month/year as direct inputs or derive from current date.
    # For simplicity, let's assume month, year, weekday, weekend, holiday are provided directly.
    # If not provided, you might want to default them (e.g., to current date's values)

    # Ensure all base features are present, fill missing with a default or error
    for feature in BASE_FEATURES:
        if feature not in input_df.columns:
            # For simplicity, fill with 0 or a reasonable default.
            # In a real app, you might want to return an error or use imputation.
            input_df[feature] = 0
            print(f"Warning: Missing feature '{feature}' in input. Filling with 0.")

    # Handle product_category_name for one-hot encoding
    # Create a new DataFrame with all expected one-hot encoded columns initialized to False
    final_input_df = input_df[BASE_FEATURES].copy() # Start with base features
    
    for col in expected_category_cols:
        final_input_df[col] = False # Initialize all dummy columns to False

    # Set the correct product category to True
    if 'product_category_name' in data and data['product_category_name']:
        category_name = data['product_category_name']
        # The column name for one-hot encoding typically follows 'prefix_value'
        # Check if the category name exists in the list of categories used for training
        if category_name in product_categories:
            # Construct the one-hot encoded column name
            # This logic must match how pd.get_dummies created columns during training
            dummy_col_name = f'product_category_name_{category_name}'
            # Ensure the dummy column exists in our expected list before trying to set it
            if dummy_col_name in final_input_df.columns:
                final_input_df[dummy_col_name] = True
            else:
                print(f"Warning: One-hot encoded column '{dummy_col_name}' not found. Category might be new or misspelled.")
        else:
            print(f"Warning: Provided product category '{category_name}' not recognized from training data.")
    
    # Ensure the order of columns in final_input_df matches the model's training features
    # This is CRUCIAL for correct predictions
    # We need to construct the full list of expected features including one-hot encoded ones
    full_expected_features = BASE_FEATURES + expected_category_cols
    
    # Reindex the DataFrame to ensure all columns are present and in the correct order
    # Fill any missing columns (e.g., new categories not seen in training) with 0/False
    final_input_df = final_input_df.reindex(columns=full_expected_features, fill_value=False)
    
    # Convert boolean columns to int (0 or 1) as scikit-learn models expect numerical input
    for col in expected_category_cols:
        if col in final_input_df.columns:
            final_input_df[col] = final_input_df[col].astype(int)

    # Check if the final input DataFrame's columns match the model's expected features
    # This is a critical validation step
    if not all(final_input_df.columns == model.feature_names_in_):
        # This indicates a mismatch in feature sets or order
        # It's good practice to log this and potentially return a more specific error
        print("Error: Feature mismatch between input data and trained model!")
        print("Input columns:", final_input_df.columns.tolist())
        print("Model expected columns:", model.feature_names_in_.tolist())
        return jsonify({'error': 'Input features do not match the trained model\'s expected features.'}), 400


    try:
        prediction = model.predict(final_input_df)
        return jsonify({'predicted_unit_price': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure the model and data files exist before running
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file '{MODEL_PATH}' not found. Please train the model and save it.")
        print("Exiting. You need to run the Jupyter Notebook to generate the .pkl file.")
        exit() # Exit if the model is not found

    if not os.path.exists(DATA_PATH):
        print(f"CRITICAL ERROR: Data file '{DATA_PATH}' not found. Please ensure 'retail_price.csv' is in the same directory.")
        print("Exiting.")
        exit() # Exit if the data is not found

    app.run(debug=True) # debug=True for development, set to False for production