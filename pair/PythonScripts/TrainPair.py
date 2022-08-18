#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrainPair

Created August 18, 2022
Last edited August 18, 2022
"""

#import tensorflow as tf
from sys import argv
import matplotlib.pyplot as plt

from DataGenPair import PairGen
from ModelPair import PairModel
from MiscFuncs import getArgsBase

def getArgs(argv):
	"""Get arguments from the command line and return them as a dictionary."""
	args = {}

	#load in the arguments by their order
	argOrder = ["modelPath", "version", "load",
				"trainGlob", "valGlob", 
				"layers1", "layers2", "filters1", "filters2",
				"distEncode", "dEarly",
				"epochs", "batchSize", "learnRate",
				"plotPath"]

	intKeys = ["version", "layers1", "layers2", "filters1", "filters2", 
				"epochs", "batchSize"]

	floatKeys = ["learnRate"]

	boolKeys = ["load", "dEarly"]
	
	return getArgsBase(argv,
						argOrder = argOrder,
						intKeys = intKeys,
						floatKeys = floatKeys,
						boolKeys = boolKeys)
	
def train(modelPath, version, load,
		  trainGlob, valGlob,
		  layers1, layers2, filters1, filters2,
		  distEncode, dEarly,
		  epochs, batchSize, learnRate,
		  plotPath):
	"""Train the model, save it, and plot the loss/accuracy over time."""
	
	#load up the data generators
	trainGen = PairGen(trainGlob, 
				batchSize = batchSize, distEncode = distEncode)
	valGen = PairGen(valGlob, shuffle = False,
				batchSize = batchSize, distEncode = distEncode)
	
	print("training and validation generators loaded.", flush = True)
	
	#make the model
	model = PairModel(layers1, layers2, filters1, filters2, 
		dEarly = dEarly, learningRate = learnRate)
	
	modelLoadPath = "{}_v{}".format(modelPath, version)
	modelSavePath = "{}_v{}".format(modelPath, version + 1)
	
	if(load):
		model.load(modelLoadPath)
		print("weights loaded from {}.".format(modelLoadPath), flush = True)
	
	#model.summary()
	
	#callbacks, to stop early
	#callbacks = tf.keras.callbacks.EarlyStopping(monitor = "val_loss",
	#				  min_delta = 1e-4,
	#				  patience = 10,
	#				  restore_best_weights = True)
	
	print("\n\n---------- Training ----------\n", flush = True)

	model.fit(trainGen,
				epochs = epochs,
				#callbacks = callbacks,
				validation_data = valGen
				)

	history = model.modelTrain.history.history
	#plot things
	fig, (lossAx, accAx) = plt.subplots(2)
	lossAx.plot(history["loss"], "#124BA1")
	lossAx.plot(history["val_loss"], "#B20F21")

	accAx.plot(history["binary_accuracy"], "#124BA1")
	accAx.plot(history["val_binary_accuracy"], "#B20F21")

	print("\n\n---------- Evaluation ----------\n", flush = True)

	evalLoss, evalMet = model.evaluate(valGen)

	print("\n\n---------- Finishing ----------\n", flush = True)

	model.save(modelSavePath)

	print("Weights saved to {}.".format(modelSavePath), flush = True)

	plt.savefig(plotPath)

	plt.clf()
	
	print("Loss & accuracy saved to {}.".format(plotPath), flush = True)

	#print out the final info
	print("binary crossentropy: {}, binary accuracy: {}".format(
		evalLoss, evalMet))

	return
	
args = getArgs(argv)
train(**args)