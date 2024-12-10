import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from speech_recognition import Recognizer, Microphone
import pandas as pd
import time
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import librosa
import numpy as np
import pyaudio
import wave

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load dataset
dataframe = pd.read_csv('voice.csv')

# Preprocess data
features = ['meanfreq', 'sd', 'median', 'IQR', 'skew', 'sp.ent', 'sfm', 'meanfun', 'minfun', 'maxfun']
inputs = dataframe[features].values

# Scale features
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)
inputs = torch.tensor(inputs, dtype=torch.float32)

label_encoder = LabelEncoder()
dataframe['label_encoded'] = label_encoder.fit_transform(dataframe['label'])  # Encode "male"/"female" to 0/1
labels = torch.tensor(dataframe['label_encoded'].values, dtype=torch.long)

# Split into train/test
train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    inputs, labels, test_size=0.2, random_state=42
)

# Create DataLoader
train_dataset = TensorDataset(train_inputs, train_labels)
test_dataset = TensorDataset(test_inputs, test_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define model
class VoiceGenderClassifier(nn.Module):
    def __init__(self):
        super(VoiceGenderClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model
model = VoiceGenderClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for batch_inputs, batch_labels in train_loader:
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        outputs = model(batch_inputs)
        _, predicted = torch.max(outputs, dim=1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
accuracy = 100 * correct / total

# Streamlit UI
st.title("Voice Gender Classification")
st.write("This application uses a deep learning model to classify gender based on voice features.")

# Accuracy Display
#st.subheader("Model Accuracy")
#st.write(f"The model achieved a test accuracy of **{accuracy:.2f}%**.")

# Audio Input for Prediction
st.subheader("Speak Into the Microphone")

# Function to record audio from microphone
def record_audio():
    p = pyaudio.PyAudio()
    
    # Set parameters
    rate = 16000  # Sample rate (Hz)
    channels = 1  # Mono
    frames_per_buffer = 1024  # Size of each audio chunk
    
    # Open a stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=frames_per_buffer)
    
    st.write("Recording... Please speak into the microphone.")
    frames = []
    for i in range(0, int(rate / frames_per_buffer * 5)):  # Record for 5 seconds
        data = stream.read(frames_per_buffer)
        frames.append(data)
    
    st.write("Recording finished.")
    
    # Stop the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the audio data into a .wav file
    filename = 'recorded_audio.wav'
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename

# Function to extract features from the audio file
def extract_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    meanfreq = np.mean(spectral_centroid)
    sd = np.std(spectral_centroid)
    median = np.median(spectral_centroid)
    IQR = np.percentile(spectral_centroid, 75) - np.percentile(spectral_centroid, 25)
    skewness = skew(spectral_centroid)
    sp_ent = np.mean(librosa.feature.spectral_flatness(y=y))
    sfm = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
    meanfun = np.mean(zero_crossing_rate)
    minfun = np.min(zero_crossing_rate)
    maxfun = np.max(zero_crossing_rate)

    # Ensure valid values for the features
    features = [meanfreq, sd, median, IQR, skewness, sp_ent, sfm, meanfun, minfun, maxfun]
    return np.nan_to_num(features)

# Button to start recording
if st.button("Start Recording"):
    audio_file = record_audio() 
    # Extract features from the recorded audio
    features = extract_features(audio_file)
    
    # Preprocess the features
    features = scaler.transform([features])  # Scale features
    input_tensor = torch.tensor(features, dtype=torch.float32)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        gender = label_encoder.inverse_transform([prediction])[0]
    
    st.write(f"Predicted Gender: **{gender}**")

# Speech-to-Text Functionality
def transcribe_audio():
    recognizer = Recognizer()
    with Microphone() as source:
        st.info("Listening... Speak into the microphone.")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("Processing audio...")
            transcription = recognizer.recognize_google(audio)
            return transcription
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            return None

# Speech-to-Text Section
st.subheader("Speech-to-Text")
if st.button("Speak into the microphone"):
    transcription = transcribe_audio()
    if transcription:
        st.write("You said:")
        st.success(transcription)
    else:
        st.warning("No speech detected or unable to transcribe.")


def normalize_feature(feature):
    return (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

# Function to detect emotions based on audio features
def detect_emotion(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract features
    pitch = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    energy = librosa.feature.rms(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    # Normalize features
    norm_pitch = normalize_feature(pitch)
    norm_energy = normalize_feature(energy)
    mean_mfcc = np.mean(mfcc[1])  # First MFCC coefficient (optional)
    
    # Compute statistics
    mean_pitch = np.mean(norm_pitch)
    mean_energy = np.mean(norm_energy)
    variance_pitch = np.var(norm_pitch)
    variance_energy = np.var(norm_energy)

    # Improved rule-based emotion detection
    if mean_energy > 0.5 and variance_pitch > 0.2:
        emotion = "Happy"
    elif mean_energy > 0.4 and variance_energy > 0.15:
        emotion = "Angry"
    elif mean_energy < 0.3 and mean_pitch < 0.3:
        emotion = "Sad"
    else:
        emotion = "Neutral"
    
    return emotion

# UI Section for Emotion Detection
st.subheader("Emotion Detection from Voice")
if st.button("Start Emotion Recording"):
    audio_file = record_audio()  # Record audio using the existing function
    emotion = detect_emotion(audio_file)
    st.write(f"Detected Emotion: **{emotion}**")






