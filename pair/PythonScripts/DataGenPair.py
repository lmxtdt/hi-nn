#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataGenPair

NEED TO CHECK THAT IT WORKS

Created August 18, 2022
Last edited August 18, 2022
"""

import numpy as np
from glob import glob
import tensorflow as tf
import csv

class PairGen(tf.keras.utils.Sequence):
	def __init__(self, npzGlob, batchSize = 32,
				shuffle = True, distEncode = "inc10"):
		self.batchSize = batchSize
		self.shuffle = shuffle
		self.distEncode = distEncode
		
		#get files
		self.npzFiles = glob(npzGlob)
		
		#load first file
		self.currFile = -1
		self.currX = None
		self.currY = None
		self.currD = None
		self.currMeta = None
		
		self.updateFile(0, firstFile = True)
		
	
	def updateFile(self, fileIdx, firstFile = False):
		self.currFile = fileIdx
		
		#load information from the NPZ file
		npzFile = np.load(self.npzFiles[fileIdx])
		
		self.currX = npzFile.get("x") #input
		self.currY = npzFile.get("y") #output
		self.currD = npzFile.get("d") #distance
		self.currMeta = npzFile.get("meta") #meta
		
		npzFile.close()
		
		#update internal stats about the file sizes
		#if this is the first file opened
		if(firstFile):
			self.updateInternal()
			print("{} files, {} samples per file".format(
				len(self.npzFiles),
				self.samplesPerFile))
		
		#adjust data as desired
		self.adjustCurr()
		
		#shuffle order
		if(self.shuffle):
			indices = np.arange(0, len(self.currY))
			np.random.shuffle(indices)
			
			self.currX = tf.gather(self.currX, indices)
			self.currY = tf.gather(self.currY, indices)
			self.currD = tf.gather(self.currD, indices)
			self.currMeta = tf.gather(self.currMeta, indices)

	def adjustCurr(self):
		#turn X and Y from bools to floats
		self.currX = self.currX.astype(float)
		self.currY = self.currY.astype(float)

		#D, the distance between the loci as an int
		#save the ones that are unlinked
		unlinked = self.currD == -1
		#change the range/encoding of the values
		if(self.distEncode == "inc1"):
			#0-1, 1 is unlinked
			self.currD = self.currD / 1000
			self.currD[unlinked] = 1
		elif(self.distEncode == "inc10"):
			#0-1, 10 is unlinked
			self.currD = self.currD / 1000
			self.currD[unlinked] = 10
		elif(self.distEncode == "dec0"):
			#1-0, 0 is unlinked
			self.currD = 1 - (self.currD / 1000)
			self.currD[unlinked] = 0
		elif(self.distEncode == "dec-1"):
			#1-0, -1 is unlinked
			self.currD = 1 - (self.currD / 1000)
			self.currD[unlinked] = -1
		else:
			raise Exception("invalid distance encoding: {}".format(self.distEncode))
		
		#add an extra dimension to D
		self.currD = np.expand_dims(self.currD, axis = 1)
				
		
	def updateInternal(self):
		self.samplesPerFile = self.currY.shape[0]

		if(self.samplesPerFile % self.batchSize != 0):
			raise Exception("batchSize is {},"
							" must be a factor of {}".format(self.batchSize, 
															 self.samplesPerFile))
		else:
			self.batchesPerFile = self.samplesPerFile // self.batchSize
			
		
	def getXYD(self, index):
		fileIdx = index // self.batchesPerFile
		batchIdx = index % self.batchesPerFile
		
		#open correct file
		if(fileIdx != self.currFile):
			self.updateFile(fileIdx)

		#get currect slices
		start = batchIdx * self.batchSize
		end = (batchIdx + 1) * self.batchSize
		batchX = self.currX[start : end]
		batchY = self.currY[start : end]
		batchD = self.currD[start : end]
		
		return batchX, batchY, batchD

	
	def __getitem__(self, index):
		batchX, batchY, batchD = self.getXYD(index)
		
		return ({"genos": batchX, "dists": batchD}, batchY)
	
	
	def __len__(self):
		return self.batchesPerFile * len(self.npzFiles)
	

	def on_epoch_end(self):
		#shuffle the order of the files of each category
		if(self.shuffle):
			np.random.shuffle(self.npzFiles)
			self.currFile = -1
			self.currX = None
			self.currY = None	
			self.currD = None
