# gan_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from sklearn.model_selection import train_test_split

# Improved Preprocess Dataset Function
def preprocess_dataset(file_path):
    try:
        # Load dataset
        data = pd.read_csv(file_path, delimiter=",", on_bad_lines="skip", dtype=str, header=None)
        
        # Ensure dataset has exactly 4 columns (Sentence, SQLInjection, XSS, Normal)
        expected_columns = 4
        data = data.dropna()
        if data.shape[1] != expected_columns:
            print(f"Error: Expected {expected_columns} columns, but found {data.shape[1]}")
            return None
        
        # Assign column names
        data.columns = ["Sentence", "SQLInjection", "XSS", "Normal"]
        
        # Convert label columns to integers
        for col in ["SQLInjection", "XSS", "Normal"]:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
        
        # Ensure Sentence column is treated as a string
        data["Sentence"] = data["Sentence"].astype(str)
        
        print("Dataset loaded and processed successfully!")
        return data
    
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None

# Load and clean dataset
data = preprocess_dataset("C:/Users/rajes/OneDrive/Desktop/gans_firewall/dataset/SQLInjection_XSS_MixDataset.1.0.0.csv")
if data is None:
    exit()

X_texts = data['Sentence'].astype(str).values
y_labels = data[['SQLInjection', 'XSS', 'Normal']].values

# Tokenization & Padding
max_words = 1000  # Vocabulary size
max_len = 100  # Max sentence length
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_texts)
X_sequences = tokenizer.texts_to_sequences(X_texts)
X_padded = pad_sequences(X_sequences, maxlen=max_len)



# Define the Generator
def build_generator(latent_dim):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(latent_dim,)),
        Dense(256, activation="relu"),
        Dense(max_len, activation="tanh")  # Generate sequences of length max_len
    ])
    return model

# Define the Discriminator
def build_discriminator():
    model = Sequential([
        Dense(256, activation="relu", input_shape=(max_len,)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(3, activation="softmax")  # Binary classification (real or fake)
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Build GAN
latent_dim = 100  # Size of the noise vector
generator = build_generator(latent_dim)
discriminator = build_discriminator()
#discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Freeze discriminator during generator training
discriminator.trainable = False

# Combine generator and discriminator into GAN
gan_input = Input(shape=(latent_dim,))
generated_sequence = generator(gan_input)
gan_output = discriminator(generated_sequence)
GAN = Model(gan_input, gan_output)
GAN.compile(loss="categorical_crossentropy", optimizer="adam")

# Training Loop
def train_gan(epochs=1000, batch_size=32):
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_padded.shape[0], batch_size)
        real_samples = X_padded[idx]
        real_labels = np.ones((batch_size, 3))  # Real data labeled as 1
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        noise = tf.convert_to_tensor(noise, dtype=tf.float32)  # Convert noise to TensorFlow tensor
        fake_samples = generator(noise, training=True)  # Use generator in training mode
        fake_labels = np.zeros((batch_size, 3))  # Fake data labeled as 0
        
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        noise = tf.convert_to_tensor(noise, dtype=tf.float32)  # Convert noise to TensorFlow tensor
        valid_labels = np.ones((batch_size, 3))  # Fake data labeled as 1 to fool discriminator
        g_loss = GAN.train_on_batch(noise, valid_labels)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Train the GAN
train_gan(epochs=100)

# Save the models and tokenizer
generator.save("models/generator.h5")
discriminator.save("models/discriminator.h5")

import pickle
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Models and tokenizer saved successfully!")