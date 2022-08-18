#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataGenFLM

Defines several data generators for the FindLociModels

NEED TO CHECK THAT IT WORKS

Created August 17, 2022
Last edited August 17, 2022
"""

import numpy as np
import tensorflow as tf
from glob import glob

class ChrGen(tf.keras.utils.Sequence):
	def __init__(self, npzGlob, batchSize = 144, shuffle = True, 
			  loadChiMeta = False):
		self.batchSize = batchSize
		self.shuffle = shuffle
		self.loadChiMeta = loadChiMeta
		
		self.npzFiles = glob(npzGlob)
		
		self.clearCurr()

		#load first file				
		self.updateFile(0, firstFile = True)
	
	def __len__(self):
		return self.batchesPerFile * len(self.npzFiles)
	
	def clearCurr(self):
		#clears all current information
		self.currFile = -1
		self.currX = None
		self.currY = None
		self.currChi = None
		self.currMeta = None

	def adjustCurr(self):
		#adjusts current X, Y, etc. as necessary
		#to be modified by any child classes
		pass
		
	def updateInternal(self):
		#update internal stats
		#i.e. samplesPerFile and batchesPerFile
		#update samplesPerFile (assumes this is the same for all files)
		self.samplesPerFile = self.currY.shape[0]

		#if batchSize is the whole file, set it to the correct number
		if(self.batchSize == "wholeFile"):
			self.batchSize = self.samplesPerFile
		
		#check that batchSize is a factor of samplesPerFile 
		if(self.samplesPerFile % self.batchSize != 0):
			raise Exception("batchSize is {},"
							" must be a factor of {}".format(self.batchSize, 
															 self.samplesPerFile))
		else:
			self.batchesPerFile = self.samplesPerFile // self.batchSize
		
	def updateFile(self, fileIdx, firstFile = False):
		#update the file by loading a different one
		self.currFile = fileIdx

		#load files from the current npz file
		npzFile = np.load(self.npzFiles[fileIdx])
		
		self.currX = npzFile.get("x")
		self.currY = npzFile.get("y")
		if(self.loadChiMeta):
			self.currChi = npzFile.get("chi")
			self.currMeta = npzFile.get("meta")
		
		npzFile.close()
		
		if(firstFile):
			#update internal stats
			self.updateInternal()
			print(("samplesPerFile: {}, batchSize: {}"
					"y dim: {}, x dim: {}").format(
						self.samplesPerFile, self.batchSize, 
						self.currY.shape, self.currX.shape))
		
		self.adjustCurr()
		
		#shuffle order
		if(self.shuffle):
			indices = np.arange(0, self.currY.shape[0])
			np.random.shuffle(indices)
			
			self.currX = tf.gather(self.currX, indices)
			self.currY = tf.gather(self.currY, indices)
			if(self.loadChiMeta):
				self.currChi = tf.gather(self.currChi, indices)
				self.currMeta = tf.gather(self.currMeta, indices)

	def getSlice(self, index):
		#helper function called by __getitem__ and getChiMeta
		#that calculates the start and end slices used for getting
		#a certain batch. It also ensures the right file is loaded
		#for that batch.

		fileIdx = index // self.batchesPerFile
		batchIdx = index % self.batchesPerFile
		
		#open correct file
		if(fileIdx != self.currFile):
			self.updateFile(fileIdx)

		#get currect slices
		start = batchIdx * self.batchSize
		end = (batchIdx + 1) * self.batchSize

		return (start, end)
	
	def __getitem__(self, index):
		start, end = self.getSlice(index)

		batchX = self.currX[start : end]
		batchY = self.currY[start : end]
		
		return batchX, batchY

	def getChiMeta(self, index):
		if(self.loadChiMeta == False):
			raise Exception("chi-squared and meta not loaded, cannot get")

		start, end = self.getSlice(index)

		batchChi = self.currChi[start : end]
		batchMeta = self.currMeta[start : end]
		
		return batchChi, batchMeta
		
	def on_epoch_end(self):
		#shuffle the order of the files of each category
		if(self.shuffle):
			np.random.shuffle(self.npzFiles)
			self.clearCurr()
