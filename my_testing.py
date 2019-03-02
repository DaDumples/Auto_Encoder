import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard
import numpy as np 
from tensorflow import set_random_seed
import os


import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)
all_images = np.zeros((train_images.shape[0],train_images.shape[1]*train_images.shape[2]))
for idx, image in enumerate(train_images):
	all_images[idx] = image.flatten()
all_images /= 255


encoder = load_model('./weights/encoder_weights.h5',compile = True)
decoder = load_model('./weights/decoder_weights.h5',compile = True)


im = 33
print(all_images[im].shape)
out1 = encoder.predict(all_images[im:im+1])
out2 = decoder.predict(out1)

plt.imshow(all_images[im].reshape(28,28),  cmap='Greys')
plt.figure()
plt.imshow(out2[0].reshape(28,28),  cmap='Greys')
plt.show()