import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


DATA_DIR = r"C:\Users\vedaa\data\landmarks"
EPOCHS = 30
BATCH_SIZE = 8


X = []
y = []
labels = sorted(os.listdir(DATA_DIR))
label_map = {label: i for i, label in enumerate(labels)}

for label in labels:
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(label_path, file))
            X.append(data)
            y.append(label_map[label])

X = np.array(X)
y = to_categorical(y, num_classes=len(labels))

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", label_map)


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential([
    LSTM(128, return_sequences=True, input_shape=X.shape[1:]),
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation="relu"),
    Dense(len(labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


os.makedirs("models", exist_ok=True)
model.save("models/sign_model.h5")

print(" MODEL TRAINING COMPLETE")
