import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard
import numpy as np 
from tensorflow import set_random_seed
import os
import sys


import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Window(QMainWindow):

	def __init__(self, latent_space, decoder, evalues, evects, means, parent = None):
		QMainWindow.__init__(self)
		self.wid = QWidget()
		self.setCentralWidget(self.wid)
		self.decoder = decoder
		self.sorted_indx = np.argsort(-evalues)
		self.evalues = evalues[self.sorted_indx]
		self.evects = evects[:,self.sorted_indx]
		self.means = means
		

		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)
		self.ax = self.figure.add_subplot(111)

		self.layout = QHBoxLayout()
		self.slider_menu = QVBoxLayout()
		self.sliders = []
		self.representation = np.zeros(latent_space)
		for i in range(latent_space):
			s = QSlider(Qt.Horizontal)
			s.setMaximum(100)
			s.setMinimum(-100)
			s.setTickInterval(1)
			s.valueChanged.connect(self.update_representation)
			self.sliders.append(s)
		for s in self.sliders:
			self.slider_menu.addWidget(s)
		self.layout.addWidget(self.canvas)
		self.layout.addLayout(self.slider_menu)

		self.wid.setLayout(self.layout)
		self.show()

	def update_representation(self):
		for i, s in enumerate(self.sliders):
			self.representation[i] = s.value()/100

		x = (self.evects@(self.representation*self.evalues).T).T + self.means
		y = self.decoder.predict(np.expand_dims(x, axis = 0))

		self.ax.clear()
		self.ax.imshow(y.reshape(28,28),  cmap='Greys')

		self.canvas.draw()

# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print(train_images.shape)
# all_images = np.zeros((train_images.shape[0],train_images.shape[1]*train_images.shape[2]))
# for idx, image in enumerate(train_images):
# 	all_images[idx] = image.flatten()
# all_images /= 255


encoder = load_model('./weights/encoder_weights.h5',compile = True)
decoder = load_model('./weights/decoder_weights.h5',compile = True)
evalues = np.sqrt(np.load(r'./weights/evalues.npy'))
evects = np.load(r'./weights/evects.npy')
means = np.load(r'./weights/means.npy')

app = QApplication(sys.argv)

main = Window(20, decoder, evalues, evects, means)
main.show()
sys.exit(app.exec_())

