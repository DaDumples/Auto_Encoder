import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import numpy as np 
from tensorflow import set_random_seed
import os


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)
all_images = np.zeros((train_images.shape[0],train_images.shape[1]*train_images.shape[2]))
for idx, image in enumerate(train_images):
	all_images[idx] = image.flatten()
all_images /= 255
all_images = all_images[:10000]


latent_space = 20

encoder_inputs = Input(shape =(all_images[0].shape))
encoder_encoded = Dense(latent_space, activation='relu')(encoder_inputs)
encoder = Model(encoder_inputs, encoder_encoded)



decoder_inputs = Input(shape =(latent_space,))
decoder_encoded = Dense(all_images[0].shape[0])(decoder_inputs)
decoder = Model(decoder_inputs, decoder_encoded)


inputs = Input(shape =(all_images[0].shape))
en_out = encoder(inputs)
de_out = decoder(en_out)
autoencoder = Model(inputs,de_out)

autoencoder.compile(optimizer='sgd', loss='mse')
autoencoder.fit(all_images, all_images, epochs=300, batch_size=10)

if not os.path.exists(r'./weights'):
    os.mkdir(r'./weights')

encoder.save(r'./weights/encoder_weights.h5')
decoder.save(r'./weights/decoder_weights.h5')
autoencoder.save(r'./weights/ae_weights.h5')

