from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.backend as K
import sys

def load_data():
    # Load the dataset from a CSV file
    df = pd.read_csv('./data/kaggle_house_data.csv')

    # Convert the 'date' column to datetime format, handling conversion errors
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Check for missing values
    print("Missing values in each column:\n", df.isnull().sum())

    # Extract the month and year from the 'date' column
    df['month'] = df['date'].apply(lambda date: date.month)
    df['year'] = df['date'].apply(lambda date: date.year)

    # Drop the 'date' column as it's no longer needed
    df = df.drop('date', axis=1)

    # Normalize or log-transform the target variable
    df['price'] = np.log1p(df['price'])  # Apply log transformation for skewed target variable

    # Separate the features (X) and the target variable (y)
    X = df.drop('price', axis=1)
    y = df['price']

    return X, y

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.1))  # Using LeakyReLU activation
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Adjusted dropout rate

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    

    model.add(Dense(1, activation='linear'))  # Output layer with 'linear' activation for regression

    # Compile the model with a lower learning rate for Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')  # Reduced learning rate
    return model

def train_model():
    X, y = load_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Reshape y_train and y_test to ensure they are 2D arrays
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Print the shapes of the training and testing data
    print("X_train shape:", X_train_scaled.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test_scaled.shape)
    print("y_test shape:", y_test.shape)

    # Check for NaN or infinite values in the datasets
    if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
        print("Warning: Training data contains NaN or infinite values")
        X_train_scaled = np.nan_to_num(X_train_scaled)

    if np.any(np.isnan(X_test_scaled)) or np.any(np.isinf(X_test_scaled)):
        print("Warning: Test data contains NaN or infinite values")
        X_test_scaled = np.nan_to_num(X_test_scaled)

    # Build the model
    model = build_model((X_train_scaled.shape[1],))

    # Use EarlyStopping to stop training when validation loss doesn't improve
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Learning rate reduction on plateau
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    model.fit(
        x=X_train_scaled,
        y=y_train,
        validation_data=(X_test_scaled, y_test),
        batch_size=29,  # Reduced batch size for potentially better generalization
        epochs=100,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    print("Model trained")

    # Convert the training history to a DataFrame
    losses = pd.DataFrame(model.history.history)

    # Plot the training losses
    losses.plot()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.close()

    # Make predictions on the test data
    predictions = model.predict(X_test_scaled)

    # Reverse the log transformation
    predictions = np.expm1(predictions)
    y_test = np.expm1(y_test)

    # Calculate and print the mean absolute error
    mae = mean_absolute_error(y_test, predictions)
    print("Mean Absolute Error:", mae)

    # Calculate and print the root mean squared error
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("Root Mean Squared Error:", rmse)

    # Calculate and print the explained variance score
    evs = explained_variance_score(y_test, predictions)
    print("Explained Variance Score:", evs)

    # Plot predictions vs actual values
    plt.scatter(y_test, predictions)
    plt.plot(y_test, y_test, 'r')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Predicted vs Actual Prices")
    plt.show()
    plt.close()

    # Calculate and plot the prediction errors
    errors = y_test - predictions

    # Plot the distribution of errors using a histogram
    sns.histplot(errors, kde=True)
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()
    plt.close()

    # Save the model and the scaler
    model.save('house_price_model.keras')

    # Clear the session to free up resources
    K.clear_session()

    print("Training and evaluation complete")
    return model, scaler

if __name__ == "__main__":
    train_model()
    sys.exit(0)