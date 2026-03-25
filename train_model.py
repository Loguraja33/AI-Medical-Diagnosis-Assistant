import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

import pickle

# Load dataset
try:
    df = pd.read_csv(r"C:\Users\thabi\Downloads\Symptom2Disease.csv")
except FileNotFoundError:
    print("Error: Symptom2Disease.csv not found")
    exit(1)

texts = df["text"]
labels = df["label"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=50)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Model
model = Sequential()

model.add(Embedding(input_dim=5000, output_dim=128, input_length=50))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(128, activation="relu"))
model.add(Dense(len(np.unique(y)), activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Early stopping
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Save model
model.save("disease_dl_model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Model trained successfully")
print(df["label"].value_counts())