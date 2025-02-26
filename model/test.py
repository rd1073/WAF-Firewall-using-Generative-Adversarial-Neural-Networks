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
data = preprocess_dataset("dataset/SQLInjection_XSS_MixDataset.1.0.0.csv")
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


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_labels, test_size=0.2, random_state=42)

# Load the trained discriminator
discriminator = tf.keras.models.load_model("models/discriminator.h5")
discriminator.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Evaluate the discriminator on the test dataset
test_loss, test_accuracy = discriminator.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# Generate synthetic data using the generator
generator = tf.keras.models.load_model("models/generator.h5")
latent_dim = 100
noise = np.random.normal(0, 1, (len(X_test), latent_dim))
generated_samples = generator.predict(noise)


# Evaluate the discriminator on the generated samples
generated_labels = np.zeros((len(X_test), 3))  # Labels for generated samples
generated_loss, generated_accuracy = discriminator.evaluate(generated_samples, generated_labels)
print(f"Generated Data Loss: {generated_loss}")
print(f"Generated Data Accuracy: {generated_accuracy}")

# Decode generated sequences back to text
def decode_sequence(sequence):
    return " ".join(tokenizer.index_word.get(idx, "") for idx in sequence if idx != 0)

# Print some generated samples
for i in range(5):
    print(f"Generated Sample {i+1}: {decode_sequence(generated_samples[i])}")

     