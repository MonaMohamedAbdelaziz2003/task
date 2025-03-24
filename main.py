import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
from google.colab import files

#
file_path = "IMDB Dataset.csv"

# قراءة البيانات
data = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
print("First 5 rows of the dataset:")
print(data.head())


if "review" not in data.columns or "sentiment" not in data.columns:
    raise KeyError("يجب أن يحتوي الملف على الأعمدة 'review' و 'sentiment'.")

texts = data["review"].astype(str).values
labels = (data["sentiment"] == "positive").astype(int).values


tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')


X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)


model = Sequential([
    Embedding(10000, 128, input_length=500),
    SimpleRNN(64, return_sequences=True),
    Dropout(0.5),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print("Training model...")
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")


model.save("sentiment_analysis_model.h5")


def predict_sentiment(text, tokenizer, model, max_length=500):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)[0][0]
    return "Positive" if prediction >= 0.5 else "Negative"


sample_text = "This movie was absolutely fantastic!"
print("Predicted Sentiment:", predict_sentiment(sample_text, tokenizer, model))
