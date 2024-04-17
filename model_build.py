import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Function to load data from directory
def load_data(directory):
    data = []
    labels = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.npy'):
            array = np.load(filepath)
            data.append(array)
            labels.append(1 if 'Make' in filepath else 0)  # 'make' -> 1, 'bail' -> 0
    return np.array(data), np.array(labels)

# Define directories
make_dir = '/Users/mfitzpatrick/Documents/Data Science/Skateboard Model/skateboard_model/Version 2/data/np_arrays/Make/'
bail_dir = '/Users/mfitzpatrick/Documents/Data Science/Skateboard Model/skateboard_model/Version 2/data/np_arrays/Bail/'

# Load data
make_data, make_labels = load_data(make_dir)
bail_data, bail_labels = load_data(bail_dir)

# Concatenate data and labels
data = np.concatenate([make_data, bail_data], axis=0)
labels = np.concatenate([make_labels, bail_labels], axis=0)

print("shape of data: ", data.shape)

# Shuffle data
indices = np.arange(len(data))
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

data = data.reshape(-1, 20, 8)

print("shape of data after reshape: ", data.shape)

# Define model
model = models.Sequential([
    layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(20, 8)),
    layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)

# Save model
model.save('binary_classification_model.h5')