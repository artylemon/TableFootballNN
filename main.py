import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# load the dataset
url = './fake_data_skewed.csv'
raw_dataset = pd.read_csv(url, na_values='?', skipinitialspace=True)

# look at the dataset
dataset = raw_dataset.copy()
print(dataset.head())

# Split the dataset into training and test sets
train_dataset = dataset.sample(frac=0.8, random_state=2)
test_dataset = dataset.drop(train_dataset.index)

# look at the dataset
print("Train dataset: ")
print(train_dataset.describe())
print()
print("Test dataset: ")
print(train_dataset.describe())

# split the features from labels
train_features = train_dataset[['li', 'si']].copy()
train_labels = train_dataset[['lo', 'so', 'ao', 'po']].copy()

test_features = test_dataset[['li', 'si']].copy()
test_labels = test_dataset[['lo', 'so', 'ao', 'po']].copy()

# normalize inputs with a normalizer layer
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))


def build_and_compile_model(norm, n_neurons, n_layers):
    model = keras.Sequential([norm])

    for i in range(n_layers):
        model.add(layers.Dense(n_neurons, activation='relu'))

    model.add(layers.Dense(1))
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.01))
    return model


def train_and_save_model(model, features, labels, name):
    loss = model.fit(
        features,
        labels,
        validation_split=0.2,
        verbose=0, epochs=500)

    model.save(name)


def test_model(model, features, labels, name):
    predictions = model.predict(features).flatten()

    error = predictions - labels
    print("Model: ", name, "\n")
    print(error.abs().describe())

    with open('out.txt', 'a') as f:
        print("Model: ", name, "\n", file=f)
        print('Filename:', error.abs().describe(), "\n", file=f)


dnn_model = build_and_compile_model(normalizer, 4, 4)

print("DNN Model: \n", dnn_model.summary(), "\n")

train_and_save_model(dnn_model, train_features, train_labels['ao'], '4-4-ao')
test_model(dnn_model, test_features, test_labels['ao'], '4-4-ao')
# dnn_model.save('dnn_model_two_layer_64')
