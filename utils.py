from tensorflow.keras.models import load_model
import joblib
import pandas as pd

def load_trained_model_and_scaler():
    model = load_model('house_price_model.keras')
    scaler = joblib.load('scaler.save')
    return model, scaler

def preprocess_single_house(house_data, scaler):
    house_df = pd.DataFrame([house_data])
    
    # Add missing features with default values if they were present during scaler fitting
    missing_features = set(scaler.feature_names_in_) - set(house_df.columns)
    for feature in missing_features:
        house_df[feature] = 0  # or another default value like None
    
    # Ensure the order of columns matches the training data
    house_df = house_df[scaler.feature_names_in_]
    
    return scaler.transform(house_df)

