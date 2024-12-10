import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Step 1: Load and Preprocess the Data
def load_and_preprocess_data():
    # Load CSV file
    data = pd.read_csv('voice.csv')

    # Display dataset preview in UI
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Features and Labels
    X = data.iloc[:, :-1].values  # All columns except the last (features)
    y = data.iloc[:, -1].values   # Last column (labels)

    # Encode categorical labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    st.write("### Label Encoding Mapping")
    st.write(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Determine the number of features
    n_features = X.shape[1]

    # Find the next perfect square
    next_square = int(np.ceil(np.sqrt(n_features))) ** 2

# Pad with zeros if necessary
    if n_features < next_square:
        padding = np.zeros((X.shape[0], next_square - n_features))
        X = np.hstack((X, padding))

    # Reshape to square dimensions
    n_side = int(np.sqrt(next_square))
    X = X.reshape(X.shape[0], n_side, n_side, 1)
    # Reshape to square dimensions suitable for Conv2D
    X = X.reshape(X.shape[0], n_features, n_features, 1)


    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, num_classes

# Step 2: Create the Model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'),  # Use padding to avoid shrinking
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train the Model
def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

# Step 4: Evaluate the Model
def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    st.write(f"### Test Loss: {loss:.4f}")
    st.write(f"### Test Accuracy: {accuracy:.4f}")

# Step 5: Visualize Training Performance
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].legend()

    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Loss')
    axes[1].legend()

    st.pyplot(fig)
    plt.close(fig)  # Prevent Streamlit rendering issues

# Step 6: Streamlit App
def main():
    st.title("Speech Recognition - Gender Classification")

    # Hyperparameters
    epochs = st.sidebar.slider("Epochs", 1, 50, 10)
    batch_size = st.sidebar.slider("Batch Size", 16, 256, 64)

    # Load and preprocess the data
    x_train, x_test, y_train, y_test, num_classes = load_and_preprocess_data()

    # Display sample shape
    st.write("### Data Shape")
    st.write(f"Input Shape: {x_train.shape}, Number of Classes: {num_classes}")

    # Create and train the model
    model = create_model(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), num_classes=num_classes)

    # Train the model
    st.subheader("Training the Model")
    with st.spinner("Training..."):
        history = train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size)
    st.success("Training Complete!")

    # Evaluate the model
    st.subheader("Evaluate the Model")
    evaluate_model(model, x_test, y_test)

    # Plot training history
    st.subheader("Training Performance")
    plot_training_history(history)

if __name__ == "__main__":
    main()
