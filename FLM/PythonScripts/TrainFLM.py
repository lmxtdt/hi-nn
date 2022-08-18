#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrainFLM: trains a model.

NEED TO CHECK THAT IT WORKS

Created August 17, 2022
Last edited August 17, 2022
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from sys import argv

from ModelFLM import FlmBuilder
from DataGenFLM import ChrGen

#note: need to adjust sys.path to get to MiscFuncs.
#or just keep a copy of it in every PythonScripts folder
#which is less elegant but just as annoying to maintain
from MiscFuncs import getArgsBase

def getArgs(argv):
	argOrder = ["trainGlob", "valGlob", "neutralGlob",
				"modelPath", "version", "load",
				"kernel1", "kernel2", "kernel3",
				"filters1", "filters2", "filters3",
				"epochs", "batchSize", "learnRate",
				"plotPath"]

	boolKeys = ["load"]

	intKeys = ["kernel1", "kernel2", "filters1", "filters2",
				"epochs", "batchSize", "version"]

	floatKeys = ["learnRate"]

	return getArgsBase(argv = argv, 
						argOrder = argOrder,
						boolKeys = boolKeys,
						intKeys = intKeys,
						floatKeys = floatKeys)

def main(trainGlob, valGlob, neutralGlob,
		 modelPath, version, load,
		 kernel1, kernel2, kernel3,
		 filters1, filters2, filters3,
		 epochs, batchSize, learnRate,
		 plotPath):
	#create the data generators
	trainGen = ChrGen(trainGlob,
					batchSize = batchSize)
	valGen = ChrGen(valGlob,
					batchSize = batchSize)
	print("training and validation generators loaded")

	#load or create the model
	if(load):
		model = tf.keras.models.load_model("{}_v{}".format(modelPath, version))
		print("model loaded")
	else:
		#normalization generator
		normGen = ChrGen(neutralGlob,
						batchSize = "wholeFile")

		#build the model
		builder = FlmBuilder(kernel1, kernel2, kernel3,
							filters1, filters2, filters3)

		builder.normalize(normGen[0][0])

		model = builder.getModel()

		#compile
		model.compile(optimizer = tf.keras.optimizers.Adam(learnRate),
					loss = tf.keras.losses.MeanSquaredError(),
					metrics = [tf.keras.metrics.MeanAbsoluteError()]
					)


		print("model created, normalized, and compiled")

	#print out model summary
	print("\n\n---------- Model ----------\n")
	model.summary()

	#configure callback (to stop early)
	callbacks = [tf.keras.callbacks.EarlyStopping(monitor = "val_loss",
											  min_delta = 1e-4,
											  patience = 10,
											  restore_best_weights = True)]

	print("\n\n---------- Training ----------\n")
	#train
	model.fit(trainGen,
		   epochs = epochs,
		   callbacks = callbacks,
		   validation_data = valGen)
	
	#make a plot of the validation and training loss
	history = model.history.history
	trainColor = "#124BA1"
	valColor = "#B20F21"
	plt.plot(history["loss"], trainColor)
	plt.plot(history["val_loss"], valColor)

	#evaluate the model
	print("\n\n---------- Evaluation ----------\n")
	
	evalLoss, evalMet = model.evaluate(valGen)
	
	#save the model, then the training figure
	print("\n\n---------- Saving ----------\n")
	
	saveModelPath = "{}_v{}".format(modelPath, version + 1)
	
	print("Saving model to {}...".format(saveModelPath))
	
	#save
	model.save(saveModelPath)
	
	plt.savefig(plotPath)
	plt.clf()
	print("Saved MSE plot to {}.".format(plotPath), flush = True)
	
	#print out the final information
	print("MSE: {}, MAE: {}".format(evalLoss, evalMet))

args = getArgs(argv)
main(**args)