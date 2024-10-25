from utils import load_trained_model_and_scaler, preprocess_single_house
import numpy as np

if __name__ == "__main__":
    # Load the trained model and scaler
    model, scaler = load_trained_model_and_scaler()

    # Define the features of a new house
    new_house_data = {
        'id': 2,
        'bedrooms': 4,
        'bathrooms': 2.5,
        'sqft_living': 1500,
        'sqft_lot': 5000,
        'floors': 2,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'grade': 8,
        'sqft_above': 2500,
        'sqft_basement': 0,
        'yr_built': 2005,
        'yr_renovated': 0,
        'zipcode': 98052,
        'lat': 47.6276,
        'long': -122.132,
        'sqft_living15': 1800,
        'sqft_lot15': 4000,
        'month': 6,
        'year': 2021
    }

    # Preprocess the new house data
    new_house_scaled = preprocess_single_house(new_house_data, scaler)

    # Make the prediction
    predicted_log_price = model.predict(new_house_scaled)

    # Reverse the log transformation
    predicted_price = np.expm1(predicted_log_price)

    # Display the predicted price
    print(f"Predicted price for the new house: ${predicted_price[0][0]:,.2f}")