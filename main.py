from model import train_model
import joblib

if __name__ == "__main__":
    # Train the model and get the scaler
    model, scaler = train_model()

    # Save the scaler
    joblib.dump(scaler, 'scaler.save')
