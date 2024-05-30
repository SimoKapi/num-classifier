import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers

IMG_WIDTH = 28
IMG_HEIGHT = 28

train = np.loadtxt("resources/mnist_train.csv", delimiter=",")
test = np.loadtxt("resources/mnist_test.csv", delimiter=",")

def main():
    train_labels, train_values = load_data(train)
    test_labels, test_values = load_data(test)

    model = get_model()
    model.fit(train_values, train_labels, epochs=4, validation_split=0.1)
    model.evaluate(test_values, test_labels, verbose=2)

    model.save("models/model.keras")

def load_data(data):
    labels = []
    lr = np.arange(10)
    for label in data[:, :1]:
        one_hot = (lr==label).astype(np.int8).round()
        labels.append(one_hot)
    labels = np.asarray(labels)

    values = (np.asfarray(data[:, 1:]) / 255).reshape(len(data), IMG_WIDTH, IMG_HEIGHT)
    return labels, values

def get_model():
    model = keras.Sequential([
        layers.Input((IMG_WIDTH, IMG_HEIGHT,1)),
        layers.Conv2D(3, kernel_size=(2, 2)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH)),
        layers.Dense(128, activation="relu"),
        layers.Dense(10)
    ])

    model.compile(
      optimizer=keras.optimizers.Adam(0.001),
      loss=keras.losses.CategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.CategoricalAccuracy()]
    )

    return model

def display_image(img, title):
    im = plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()