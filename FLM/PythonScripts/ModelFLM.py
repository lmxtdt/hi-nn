#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelFLM: Defines the construction of an FLM model

Created August 17, 2022
Last edited August 17, 2022
"""
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.layers.experimental.preprocessing \
		import Normalization as layerNorm

class FlmBuilder:
	def __init__(self, kernel1, kernel2, kernel3,
					filters1, filters2, filters3):
		numA = 2
		numB = 2
		numC = 2

		#define the model
		self.layers = []

		#normalization layer
		self.layers.append(layerNorm(name = "normalization",
									input_shape = (1000, 4)))
		#convolutional layers (block A)
		for i in range(numA):
			self.layers.append(
				layers.Conv1D(filters1, kernel1,
					padding = "same", activation = "relu",
					name = "conv1d_a{}".format(i + 1)))
		#average pooling
		self.layers.append(
			layers.AveragePooling1D(5,
				padding = "same",
				name = "avgPool1d"))
		#convolutional layers (block B)
		for i in range(numB):
			self.layers.append(
				layers.Conv1D(filters2, kernel2,
					padding = "same", activation = "relu",
					name = "conv1d_b{}".format(i + 1)))
		#max pooling
		self.layers.append(
			layers.MaxPool1D(2,
				padding = "same",
				name = "maxPool_1d"))
		#convolutional layers (block C)
		for i in range(numC):
			self.layers.append(
				layers.Conv1D(filters3, kernel3,
					padding = "same", activation = "relu",
					name = "conv1d_c{}".format(i + 1)))
		#final layers
		self.layers.append(
			layers.Conv1D(1, kernel3,
				padding = "same", activation = "sigmoid",
				name = "conv1d_out"))
		self.layers.append(
			layers.Flatten(name = "flatten"))

		#all layers have been created, create the model
		self.model = Sequential(self.layers)

	def normalize(self, neutralData):
		#normalize the normalization layer
		norm = self.model.get_layer("normalization") #check!
		norm.adapt(neutralData)

	def getModel(self):
		return self.model