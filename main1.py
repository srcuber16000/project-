import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2

training_data = []
DATADIR = "C:/Users/Asus/Downloads/archive/kagglecatsanddogs_3367a/PetImages"
CATEGORIES = ["Dog", "Cat"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            if img_array is not None:
                new_array = cv2.resize(img_array, (50, 50))
                training_data.append(new_array)
        except Exception as e:
            pass

training_data = np.array(training_data)

training_data = training_data / 255.0

train_images, test_images = train_test_split(training_data, test_size=0.2)


model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Reshape((12, 12, 1)),
    keras.layers.Conv2DTranspose(64, (3, 3), activation='relu'),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.Conv2DTranspose(64, (3, 3), activation='relu'),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')


model.fit(train_images, train_images, epochs=10, batch_size=32, validation_data=(test_images, test_images))


decoded_images = model.predict(test_images)


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i].reshape(50, 50), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(50, 50), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

