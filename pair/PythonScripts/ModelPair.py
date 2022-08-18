#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelPair
Defines a Tensorflow model trained to classify pairs
Technically, the PairModel class is not a Tensorflow model, but rather
contains two of them. One for training on fixed-length input, one for
testing on variable-length input

Created August 18, 2022
Last edited August 18, 2022
"""
import tensorflow as tf
import tensorflow.keras.layers as layers

#symmetric function that summarizes the distribution of the features of
#all individuals by the mean and variance of each feature
#standard deviation makes the weights turn into nan, for some reason
#which is why variance is uded instead of standard dev.
def VarMeanFunc(inputs):
	var = tf.math.reduce_variance(inputs, axis = 1)
	means = tf.math.reduce_mean(inputs, axis = 1)
	
	#concatenate the two of them together
	joined = tf.concat([var, means], axis = 1)
	
	return joined

#Class that contains the model that classifies pairs
#One for training on fixed-length data
#One for predicting on variable-length data
class PairModel:
	def __init__(self, layers1 = 1, layers2 = 2, filt1 = 32, filt2 = 32, 
				learningRate = 0.001, dEarly = True):
		#loss
		self.loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)
		#metrics
		self.metrics = ["binary_accuracy"]
		
		self.layers1 = layers1
		self.layers2 = layers2
		self.filt1 = filt1
		self.filt2 = filt2
		
		self.learningRate = learningRate
		self.dEarly = dEarly
		
		self.createLayers()
		self.createTrainModel()
		self.createEvalModel()
	
	
	def createLayers(self):
		#for training model
		#before the symmetric layer
		self.layersA = []
		for i in range(self.layers1):
			self.layersA.append(layers.Dense(self.filt1,
										  activation = "relu",
										  name = "dense_a{}".format(i)))

		#symmetric layer
		self.layerSymm = layers.Lambda(VarMeanFunc, name = "symmFunc")

		#merge in the distance (d)
		self.layerConcatD = layers.Lambda(lambda x: tf.concat(x, axis = 1),
									name = "concatD")
		
		#after the symmetric layer
		self.layersB = []
		for i in range(self.layers2):
			self.layersB.append(layers.Dense(self.filt2,
										activation = "relu",
										name = "dense_b{}".format(i)))
		#final output
		self.layerOut = layers.Dense(1,
							   activation = "sigmoid",
							   name = "output")
		
		#additional layers for variable length input
		#split input when it's ragged
		self.layerSplit = layers.Lambda(lambda x: x.row_splits,
										  name = "splits")
		self.layerFlat = layers.Lambda(lambda x: tf.concat(x.flat_values, 
														  axis = 0),
											  name = "flatten")
		#merge previously split info
		self.layerMerge = layers.Lambda(lambda x: tf.RaggedTensor.from_row_splits(
												values = x[0], row_splits = x[1]),
											name = "merge")


	def createTrainModel(self):
		#create input
		trainInput = tf.keras.Input(shape = (500, 6), 
							  dtype = tf.float64,
							  name = "genos")
		trainD = tf.keras.Input(shape = 1, 
						  dtype = tf.float64,
						  name = "dists")
		
		#run through all layers
		#input
		x = trainInput
		#block A layers
		for layer in self.layersA:
			x = layer(x)
		#symmetric function
		x = self.layerSymm(x)
		#block B layers & additional input (distance)
		if(self.dEarly):
			x = self.layerConcatD([x, trainD])
			for layer in self.layersB:
				x = layer(x)
		else:
			for layer in self.layersB:
				x = layer(x)
			x = self.layerConcatD([x, trainD])
		#output
		trainOutput = self.layerOut(x)
		
		#create the model
		self.modelTrain = tf.keras.Model(inputs = [trainInput, trainD], 
								   outputs = trainOutput)
		
		#compile the model
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learningRate)
		self.modelTrain.compile(optimizer = self.optimizer,
								loss = self.loss,
								metrics = self.metrics)
	
	
	def createEvalModel(self):
		#create input
		evalInput = tf.keras.Input(shape = (None, 6), 
							  dtype = tf.float64,
							  ragged = True,
							  name = "genos")
		evalD = tf.keras.Input(shape = 1, 
						 dtype = tf.float64, 
						 name = "dists")
		
		#get splits and flattened version of x
		splits = self.layerSplit(evalInput)
		x = self.layerFlat(evalInput)
		
		#run through all layers
		#block A layers
		for layer in self.layersA:
			x = layer(x)
		#merge before the symmetric functions
		x = self.layerMerge((x, splits))
		#symmetric function
		x = self.layerSymm(x)
		#block B layers & additional input (D)
		if(self.dEarly):
			x = self.layerConcatD([x, evalD])
			for layer in self.layersB:
				x = layer(x)
		else:
			for layer in self.layersB:
				x = layer(x)
			x = self.layerConcatD([x, evalD])


		#output
		evalOutput = self.layerOut(x)
		
		#create the model
		self.modelEval = tf.keras.Model(inputs = [evalInput, evalD], 
								  outputs = evalOutput)
		
		#compile the model
		self.sgdOptimizer = tf.keras.optimizers.SGD(learning_rate = 0.0)
		self.modelEval.compile(optimizer = self.sgdOptimizer,
								 loss = self.loss,
								 metrics = self.metrics)
	
	#basic model stuff
	def summary(self, trainModel = True):
		if(trainModel):
			return self.modelTrain.summary()
		else:
			return self.modelEval.summary()
		
	def fit(self, *args, **kwargs):
		return self.modelTrain.fit(*args, **kwargs)
	
	def evaluate(self, *args, ragged = False, **kwargs):
		if(ragged):
			return self.modelEval.evaluate(*args, **kwargs)
		else:
			return self.modelTrain.evaluate(*args, **kwargs)
		
	def predict(self, *args, ragged = False, **kwargs):
		if(ragged):
			return self.modelEval.predict(*args, **kwargs)
		else:
			return self.modelTrain.predict(*args, **kwargs)
	
	#loading and saving
	def compatible(self, loadedModel):
		#determines if loadedModel's configuration is compatible with this
		#instance's
		
		#check that they have the same input shape
		if(loadedModel.layers[0].input_shape != self.modelTrain.layers[0].input_shape):
			return (False, "Trying to load model with input shape {} "
				   "into a model with input shape {}".format(
									      loadedModel.layers[0].input_shape,
										  self.modelTrain.layers[0].input_shape))
		
		#check they have the same number of layers
		if(len(loadedModel.layers) != len(self.modelTrain.layers)):
			return (False, "Trying to load model with {} layers "
				   "into a model with {} layers".format(
					   len(loadedModel.layers),
					   len(self.modelTrain.layers)))
			
		#compare the name, types, and units of the layers
		#yes, they must be in the same order
		for i in range(len(loadedModel.layers)):
			il = loadedModel.layers[i]
			it = self.modelTrain.layers[i]
			#names
			if(il.name != it.name):
				return (False, "Layer {}, names {} and {} not the same".format(
					i,
					il.name,
					it.name))
			#types
			if(type(il) != type(it)):
				return (False, "Layer {}, types {} and {} not the same".format(
					i,
					type(il),
					type(it)))
			#check the right number of units for a Dense layer
			if(type(il) == tf.keras.layers.Dense):
				if(il.units != it.units):
					return (False, "Layer {}, units {} and {} not equal".format(
						i,
						il.units,
						it.units))
			#expand checks as necessary
				
		return (True, "")
			
	def save(self, path):
		self.modelTrain.save(path)
		
	def load(self, path):
		#overwrites this model's architecture with the other's
		#first, load the other model
		loadedModel = tf.keras.models.load_model(path)
		
		layerNames = []
		for layer in loadedModel.layers:
			layerNames.append(layer.name)
		
		#go through layersA block
		numA = 0
		layersA = []
		while(True):
			layerName = "dense_a{}".format(numA)
			if(layerName in layerNames):
				numA += 1
				layersA.append(loadedModel.get_layer(layerName))
			else:
				break
		
		#find number of filters
		if(numA == 0):
			filtA = -1
		else:
			filtA = loadedModel.get_layer("dense_a0").units
			
		#figure out how  many layers are in the layersB block
		numB = 0
		layersB = []
		while(True):
			layerName = "dense_b{}".format(numB)
			if(layerName in layerNames):
				numB += 1
				layersB.append(loadedModel.get_layer(layerName))
			else:
				break
			
		#find number of filters
		if(numB == 0):
			filtB = -1
		else:
			filtB = loadedModel.get_layer("dense_b0").units
			
		#determine dEarly based off the position of the layers
		#"dists" and "symmFunc"
		symmIdx = layerNames.index("symmFunc")
		distIdx = layerNames.index("dists")
		if(distIdx == symmIdx + 1):
			dEarly = True
		else:
			dEarly = False
			
		#save the new configuration
		self.layers1 = numA
		self.layers2 = numB
		self.filt1 = filtA
		self.filt2 = filtB
		self.dEarly = dEarly
		
		#use the new configuration to create inner layers
		self.createLayers()
		
		#replace layersA with the loadedModel's layersA
		#etc. for layersB
		#(createLayers was called to configure the shape of the symmetric
		#function and similar layers)
		self.layersA = layersA
		self.layersB = layersB
		
		#save the remaining layers and model
		self.layerOut = loadedModel.get_layer("output")
		self.modelTrain = loadedModel
		
		#create other model
		self.createEvalModel()

		
	def loadCompatible(self, path):
		#checks that the other model has compatible architecture
		#first, load the other model
		loadedModel = tf.keras.models.load_model(path)
		
		#check that they have compatible architectures
		compatible, msg = self.compatible(loadedModel)
		if(not compatible):
			raise Exception("Error in loading model: {}".format(msg))
			
		#update internal layers
		newLayersA = []
		for lA in self.layersA:
			newLayersA.append(loadedModel.get_layer(name = lA.name))
		self.layersA = newLayersA
		
		newLayersB = []
		for lB in self.layersB:
			newLayersB.append(loadedModel.get_layer(name = lB.name))
		self.layersB = newLayersB
			
		self.layerOut = loadedModel.get_layer(name = "output")
		
		#update the training model
		self.modelTrain = loadedModel

		#re-make the evaluation model
		self.createEvalModel()
	
	#note: saveWeights & loadWeights don't work for some reason.
	def saveWeights(self, path):
		self.modelTrain.save_weights(path, save_format = "tf")
	
	def loadWeights(self, path):
		self.modelTrain.load_weights(path, by_name = True)
		#re-compile the other model
		self.createEvalModel()