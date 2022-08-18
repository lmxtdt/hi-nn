#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProcDataPair

Extracts all pairs from a simulations and an equal number of false pairs.

NEED TO CHECK IT WORKS, for both ragged & not

Created August 17, 2022
Last edited August 17, 2022
"""
from sys import argv
import numpy as np
import csv
from scipy.stats import chisquare
from scipy.signal import find_peaks
from IndDataCompression import decompress

class IndAncestry:
	def __init__(self, indPath, posPath, seed):
		self.indPath = indPath
		self.posPath = posPath
		self.seed = seed

		#other parameters
		self.numInds = 500 #number of individuals to return

		#do important functions
		self.loadFiles()
		self.pickLoci()
		self.pickPairs()
		self.gatherLoci()

	def loadFiles(self):
		#read compressed individual ancestry
		self.ind = decompress(self.indPath)

		#read positions
		posRows = []
		#open file, which contains the positions
		#of incompatibility pairs (for now)
		with open(self.posPath, "r") as file:
			reader = csv.reader(file)
			for row in reader:
				posRows.append(row)
		#save positions
		self.pos = np.array(posRows, dtype = int)

		#save number of individuals
		self.n = len(indRows)

	def pickLoci(self):
		#to find loci not involved in incompatibilities
		#calculate chisquared values over genome
		#and pick regions with the lowest p-values

		#calculate chi-squared
		sumHomo1 = np.sum(self.ind == 0, axis = 0)
		sumHet = np.sum(self.ind == 1, axis = 0)
		sumHomo2 = np.sum(self.ind == 2, axis = 0)
		genos = np.stack([sumHomo1, sumHet, sumHomo2])

		expected = np.reshape([0.25, 0.5, 0.25], (3, 1)) * self.n

		chi = chisquare(genos, expected, axis = 0).pvalue

		#find local minima in p-value & sort by p-value
		minima = []
		heights = []
		#for each chromosome
		for i in range(self.ind.shape[1] // 1000):
			minimum = find_peaks(- chi[(i * 1000):((i + 1) * 1000)],
				height = -0.5, distance = 250)
			minima.append(minimum[0] + (i * 1000))
			heights.append(minimum[1]["peak_heights"])

		allMin = np.concatenate(minima)
		sortedMin = allMin[np.argsort(np.concatenate(heights))]

		#make sure they are far enough away from actual incompatibilities
		#here, "far enough" being defined as at least 25 map units away
		colPos = np.reshape(self.pos, (1, -1))
		rowMin = np.reshape(sortedMin, (-1, 1))

		#to avoid thinking two peaks on different chromosomes are close
		#add 100,000 units between each chromsome
		sepPos = ((colPos // 1000) * 100000) + colPos
		sepMin = ((rowMin // 1000) * 100000) + rowMin

		#determine the distance between the incompatibilities
		#and chi-squared minima
		#and select only chi-squared loci at least 25 mu away
		#from real incompatibilities.
		dist = np.abs(sepPos - sepMin)
		minDist = np.min(dist, axis = 1)
		farEnough = minDist > 250
		filtMin = sortedMin[farEnough]

		#pick the most distorted of the filtered chi-squared p-value minima
		#to be the fake/false loci
		numFake = self.pos.size
		
		if(len(filtMin) < numFake):
			self.chosenFake = np.random.choice(filtMin, 
												size = numFake, replace = True) 
		else:
			self.chosenFake = np.array(filtMin[-numFake:])

		np.random.shuffle(self.chosenFake)

		#pick real positions
		np.random.shuffle(self.pos)
		self.chosenReal = self.pos

	def pickPairs(self):
		#record the valid pairs, i.e. actual incompatibility pairs
		#by adding the pairs, adding the flipped pairs, and
		#labeling them all with 1
		validPairs = np.concatenate([np.concatenate([self.chosenReal,
										np.flip(self.chosenReal, axis = 1)]),
									np.ones((self.chosenReal.size, 1), dtype = int)],
									axis = 1)

		#split the loci into groups 1 & 2
		#divide real pairs so that both members of each pair are in the same
		#group, and will not be paired with each other
		grp1NumValid = self.chosenReal.shape[0] // 2
		grp1Valid = np.reshape(self.chosenReal[0:grp1NumValid], -1)
		grp2Valid = np.reshape(self.chosenReal[grp1NumValid:], -1)
				
		#select the false loci
		grp1NumFalse = self.chosenReal.size - grp1Valid.size
		grp1False = self.chosenFake[0:grp1NumFalse]
		grp2False = self.chosenFake[grp1NumFalse:]
		
		#join the groups and shuffle them
		grp1 = np.concatenate((grp1Valid, grp1False))
		grp2 = np.concatenate((grp2Valid, grp2False))
		np.random.shuffle(grp1)
		np.random.shuffle(grp2)
				
		#pair each member of group 1 with a member of group 2
		#by stacking them together 
		groupsPaired = np.stack((grp1, grp2), axis = 1)
		#randomly halve them, so the numbers are right
		groupsHalved = groupsPaired[0:(groupsPaired.shape[0] // 2)]
		#record the invalid pairs in the same way as with the valid pairs
		#i.e. flipping them, labeling with 0
		invalidPairs = np.concatenate([np.concatenate([groupsHalved,
												 np.flip(groupsHalved, axis = 1)]),
								 np.zeros((groupsHalved.size, 1), dtype = int)],
								axis = 1)

		#join all pairs that will be used
		self.chosenPairs = np.concatenate([validPairs, invalidPairs])
		
		#determine if they're on the same chromosome
		difChr = (self.chosenPairs[:, 0] // 1000) != (self.chosenPairs[:, 1] // 1000)
		distances = np.abs(self.chosenPairs[:, 0] - self.chosenPairs[:, 1])
		distances[difChr] = -1
		self.dist = distances

	def pickInds(self):
		#decide which individuals to gather
		indIds = np.arange(0, self.n)
		#if fewer than 500 individuals, need to sample multiple times
		if(self.n < self.numInds):
			#if < 250 individuals, sample completely randomly with replacement
			if(self.numInds // 2):
				chosenInds = np.random.choice(indIds, size = self.numInds, 
												replace = True)
			#if >= 250 individuals, sample all individuals once and some twice
			else:
				chosenInds = np.concatenate([indIds,
					np.random.choice(indIds, size = self.numInds - self.n, 
									replace = False)])
		#if there are at least 500 individuals, pick them randomly
		else:
			chosenInds = np.random.choice(indIds, size = self.numInds, 
											replace = False)
		#select the correct individuals and then transpose so the shape is
		#(locus, invididual)
		self.indGenos = np.transpose(self.ind[chosenInds,])

	def gatherLoci(self):
		#select the correct loci in the chosen individuals
		locs1 = self.indGenos[self.chosenPairs[:, 0]]
		locs2 = self.indGenos[self.chosenPairs[:, 1]]

		#one-hot encode the genotypes
		code = {0: [True, False, False], #homozygous parent 1
				1: [False, True, False], #heterozygous
				2: [False, False, True]} #homozygous parent 2
		oneHot = lambda row : np.array([code[x] for x in row])

		hot1 = np.apply_along_axis(oneHot, axis = 1, arr = locs1)
		hot2 = np.apply_along_axis(oneHot, axis = 1, arr = locs2)

		#join loci 1 & 2 together at the last axis
		#so the last dimension is 6 long, containing the
		#one-hot encoding for both loci
		hotArr = np.concatenate([hot1, hot2], axis = 2)

		#save x & y & d
		#make sure y is also a bool, for space
		#make sure d is a int16, for space (int8 is too small)
		#note: this relies on chromosomes being 1000 long
		#maximum distance allowed is ~32k (32,767)
		self.x = hotArr
		self.y = self.chosenPairs[:, 2].astype(bool)
		self.d = self.dist.astype(np.int16)


	def getXYD(self):
		return (self.x, self.y, self.d)


	def getMeta(self):
		#meta is:
		#the simulation seed
		#the position of loc A
		#and the position of loc B
		stacked = np.stack([np.repeat(self.seed, self.chosenPairs.shape[0]),
							self.chosenPairs[:,0],
							self.chosenPairs[:,1]],
							axis = 1)
		return stacked

class IndAncestryRagged(IndAncestry):
	def pickInds(self):
		self.indGenos = np.transpose(self.ind)

######

inputCSV = argv[1]  #name of the CSV file that contains all the relevant info.
outName = argv[2]   #name of file to write the .npz to
ragged = bool(int(argv[3])) #whether to use all individuals rather than 500

print("reading {} and outputting to {}".format(inputCSV, outName), 
	  flush = True)

xArr = []
yArr = []
dArr = []
mArr = []
breaks = [0]

with open(inputCSV, "r") as cfile:
	reader = csv.reader(cfile)

	#each row is one simulation, and contains
	#seed (int); ancPopOutPath (str); ancIndOutPath (str); 
	#saveIndData (boolean); fitOutPath (str); posOutPath (str);
	#N (int); num. F2s alive (int); chrL (int); 
	#numChr (int); numPairs (int); probIntra (float); 
	#genoDec (float)
	for row in reader:
		#read the row
		seed = int(row[0])
		indPath = row[2]
		posPath = row[5]

		print("analyzing {}...".format(indPath), flush = True)

		#calculate summary stats & output
		if(ragged):
			am = IndAncestryRagged(indPath, posPath, seed)
		else:
			am = IndAncestry(indPath, posPath, seed)
		x, y, d = am.getXYD()
		meta = am.getMeta()

		#append to arrays
		xArr.extend(x)
		yArr.extend(y)
		dArr.extend(d)
		mArr.extend(meta)

		#add the number of individuals once for each pair
		#represented in the input
		for i in range(y.size):
			breaks.append(x.shape[1] + breaks[-1])

		
print("writing to {}...".format(outName))

#save all input and outputs to a compressed .npz file
if(ragged):
	#concatenate xArr together and save the breaks
	#for a ragged file
	np.savez_compressed(outName,
		  	x = np.concatenate(xArr, axis = 0),
			y = np.array(yArr),
			d = np.array(dArr),
			meta = np.array(mArr),
			breaks = breaks)
else:
	np.savez_compressed(outName,
			  	x = np.array(xArr),
				y = np.array(yArr),
				d = np.array(dArr),
				meta = np.array(mArr))

print("done")