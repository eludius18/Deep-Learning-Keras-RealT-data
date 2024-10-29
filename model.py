import os
import sys
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, ReLU
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K

# Disable macOS hardened runtime and GPU usage
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def load_data(filepath='./data/kaggle_house_data.csv'):
    """Loads and preprocesses dataset."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print("Missing values in each column:\n", df.isnull().sum())
    
    # Extract month and year from 'date' and drop 'date' column
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df.drop('date', axis=1, inplace=True)

    # Apply log transformation to 'price' and split into features/target
    df['price'] = np.log1p(df['price'])
    X, y = df.drop('price', axis=1), df['price']
    return X, y

def build_model(input_shape, neurons=64, layers=3, dropout_rate=0.2, 
                learning_rate=0.001, activation='leaky_relu'):
    """Builds a Keras model with specified hyperparameters."""
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    for _ in range(layers):
        model.add(Dense(neurons))
        model.add(LeakyReLU(negative_slope=0.1) if activation == 'leaky_relu' else ReLU())
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))  # Regression output

    # Compile model with optimizer and loss
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def train_model():
    """Main training function including hyperparameter search and evaluation."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Wrap model with KerasRegressor
    model = KerasRegressor(model=build_model, input_shape=X_train_scaled.shape[1], verbose=0)
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'model__neurons': [32, 64],
        'model__layers': [2, 3],
        'model__dropout_rate': [0.2, 0.3],
        'model__learning_rate': [0.001],
        'model__activation': ['relu'],
        'epochs': [100],
        'batch_size': [16]
    }

    
    # Execute GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    # Train model with best parameters
    best_model = build_model(input_shape=X_train_scaled.shape[1],
                             neurons=best_params['model__neurons'],
                             layers=best_params['model__layers'],
                             dropout_rate=best_params['model__dropout_rate'],
                             learning_rate=best_params['model__learning_rate'],
                             activation=best_params['model__activation'])

    history = best_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model
    predictions = best_model.predict(X_test_scaled)
    predictions = np.expm1(predictions)
    y_test = np.expm1(y_test)
    mae, rmse, evs = mean_absolute_error(y_test, predictions), np.sqrt(mean_squared_error(y_test, predictions)), explained_variance_score(y_test, predictions)
    
    print("\n--- Best Hyperparameters ---")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error: {mae}\nRoot Mean Squared Error: {rmse}\nExplained Variance Score: {evs}")

    best_model.save('house_price_model.keras')
    K.clear_session()
    return best_model, scaler

if __name__ == "__main__":
    train_model()
    sys.exit(0)