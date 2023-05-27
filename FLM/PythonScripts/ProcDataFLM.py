#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process simulation information and writes to two output files:
	an .npz file that contains X, Y, and meta used by the evaluation scripts
	a .csv file that contains additional meta about each incompatibility locus
	
Created August 5, 2022
Last edited August 19, 2022
"""

from sys import argv

import numpy as np
from scipy.stats import chisquare

import csv

class AncestryMatrix:
	#calculates and stores a matrix of the ancestry of all F2 individuals
	#no previous generations
	
	def __init__(self, ancNPZ, numChr, seed):
		self.seed = seed
		self.numChr = numChr
        
		npzFile = np.load(ancNPZ)
		ancArray = npzFile.get("pop")
		npzFile.close()
		
		totalLength = ancArray.shape[1]
						
		#add data to self
		self.chrLength = totalLength // numChr
		self.nInds = np.sum(ancArray[:, 0])
		
		self.sumHomo1 = ancArray[0]
		self.sumHet = ancArray[1]
		self.sumHomo2 = ancArray[2]
		
		#X
		
		self.meanHomo1 = self.sumHomo1 / self.nInds
		self.meanHet = self.sumHet / self.nInds
		self.meanHomo2 = self.sumHomo2 / self.nInds
		
		#note: meanAnc is specifically ancestry for p2
		self.meanAnc = (self.sumHomo2 + (self.sumHet * 0.5)) / self.nInds
		
		self.chi = np.full(totalLength, -1.0)
		
		#Y
		
		self.y = None
		
		#chromosome number
		self.realChr = []
		self.realLoc = []
		self.realChrPartner = []
		self.realLocPartner = []
		self.selec = []
		self.genos = []
		self.genoIdx = []
		#location,
		#chromosome of partner
		#location of partner
		#selective coefficient
		#architecture
		
		self.calcAllStats()
	
	def calcAllStats(self):
		#reformat the genotypes to work better with the functions
		#so each row is the genotypes for a particular locus
		actualSum = np.transpose([self.sumHomo1, self.sumHet, self.sumHomo2])
		
		#punExpect = np.array([0.25, 0.5, 0.25])
		
		#get expected frequencies
		#based off Hardy-Weinberg equilibrium
		#where p = p2 ancestry and q = p1 ancestry
		p = self.meanAnc
		q = 1 - p
		
		#make an array of expected frequencies of each genotype
		calcPqExpect = np.frompyfunc(lambda p, q: np.array([p ** 2, 2 * p * q, q ** 2]),
									 2, 1)
		pqExpect = calcPqExpect(p, q)
				
		#go through every position on the chromosome to calculate
		#chisquared likelihood of the actual frequencies of genotypes
		for i in range(self.chrLength * self.numChr):
			_, self.chi[i] = chisquare(actualSum[i],
								  pqExpect[i] * self.nInds)
		
	def calcYLoc(self, f, p):
		dec = 1.5
		
		y = np.zeros(int(self.chrLength / 10) * self.numChr)
		
		#for each incompatibility
		for i in range(p.shape[0]):
			#calculate which genotypes are affected
			#and compress into a single number, which represents a series
			#of bits, one for each genotype that could be affected
			thisGenos = f[:, i]
			genoCode = 0
			for j in range(thisGenos.shape[0]):
				genoCode += (2 ** j) * (thisGenos[j] < 1)
			
			#flatten all the positions and go through all of them
			positions = np.reshape(p[i], -1)

			for k in range(positions.shape[0]):
				#add the incompatibility to the appropriate bin in y
				#with a value equal to the selective coefficient
				y[int(positions[k] / 10)] = 1.0 - np.min(thisGenos)
				
				#fill in the meta info
				#location of this locus
				self.realChr.append(positions[k] // self.chrLength)
				self.realLoc.append(positions[k] % self.chrLength)
				
				#locations of this locus's partners
				partners = positions[positions != positions[k]]				
				self.realChrPartner.append(partners // self.chrLength)
				self.realLocPartner.append(partners % self.chrLength)
				
				#selective coefficient
				self.selec.append(np.min(thisGenos))
				#which genotypes are affected by the incompatibility
				self.genos.append(genoCode)
				#and which locus this one counts as in the above calculation
				self.genoIdx.append(k)
						
		#break y into chromosomes
		split = np.split(y, self.numChr, axis = 0)
		
		#for each chromosome
		for i in range(len(split)):
			#transform the peak heights into proper peaks
			#first go forward along the chromosome
			for j in range(split[i].shape[0] - 1):
				split[i][j + 1] = max(split[i][j + 1],
									  split[i][j] / dec)
			#then backward
			for j in range(1, split[i].shape[0]):
				split[i][-j - 1] = max(split[i][-j - 1],
								   split[i][-j] / dec)
				
		self.y = split
	
		
	def getX(self):
		#reformat compressed stats into long stats
		stacked = np.stack([self.meanAnc, 
							self.meanHomo1, self.meanHet, self.meanHomo2], axis = 1)
		#split into 
		split = np.split(stacked, self.numChr, axis = 0)
		return split
	
	def getY(self):
		return self.y
		
	def getChi(self):
		split = np.split(self.chi, self.numChr, axis = 0)
		return split
	
	def getChrMeta(self):
		#meta that the prediction matrix will use
		stacked = np.stack([
			np.repeat(self.seed, self.numChr),    #seed (sim ID)
			np.arange(self.numChr, dtype = int)]) #chromosome ID
		
		return np.transpose(stacked)
	
	def getRealMeta(self):
		#stack together most of the per-incompatibility meta info
		stacked = np.stack([
			np.repeat(self.seed, len(self.realChr)),  #seed
			np.repeat(self.nInds, len(self.realChr)), #N
			self.selec,       #selective coefficient
			self.genos,       #architecture, i.e. which genotypes are affected
			self.genoIdx,     #which locus this one is in calculating the above
			self.realChr,     #chr number
			self.realLoc      #location
			])
		#add the partner info
		concatenated = np.concatenate([
			stacked,
			np.transpose(self.realChrPartner),
			np.transpose(self.realLocPartner)])
		#return the total, transposed
		#so each row is a real locus
		#and not a metric
		return(np.transpose(concatenated))

inputCSV = argv[1]  #name of the CSV file that contains all the relevant info.
outNPZ = argv[2]    #name of file to write the .npz to
outCSV = argv[3]    #name of file to write .csv to, of the meta info

print("analyzing {} and outputting to {} (NPZ) and {} (meta CSV)".format(
	inputCSV, outNPZ, outCSV), 
	  flush = True)

xArr = []
yArr = []
mArr = []
cArr = []
metaOut = []

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
		popPath = row[1]
		#modify population path so it ends in .npz rather than .csv
		popPath = popPath.rsplit(".", 1)[0] + ".npz"
		print("analyzing {}...".format(popPath), flush = True)
		#ancIndName = row[2]
		#saveIndData = row[3]
		fitName = row[4]
		posName = row[5]
		numChr = int(row[9])
		
		#get fitness information
		f = []
		with open(fitName, "r") as fitFile:
			reader = csv.reader(fitFile)
			for row in reader:
				f.append(row)
		f = np.array(f).astype(float)
		
		#get position information
		p = []
		with open(posName, "r") as posFile:
			reader = csv.reader(posFile)
			for row in reader:
				p.append(row)	
		p = np.array(p).astype(int)
		
		#read the tree, get stats
		am = AncestryMatrix(popPath, numChr, seed)
		am.calcYLoc(f, p)
		x = am.getX()
		y = am.getY()
		chi = am.getChi()
		meta = am.getChrMeta()
		realMeta = am.getRealMeta()

		#append to arrays
		xArr.extend(x)
		yArr.extend(y)
		mArr.extend(meta)
		cArr.extend(chi)
		metaOut.append(realMeta)
		
#save all input and outputs to a compressed .npz file
np.savez_compressed(outNPZ,
		  	x = np.array(xArr),
			y = np.array(yArr),
			chi = np.array(cArr),
			meta = np.array(mArr))

#save meta information to a (not compressed) csv file
metaArr = np.concatenate(metaOut)

with open(outCSV, "a") as file:
	np.savetxt(file, metaArr,
			   fmt = "%i,%i,%f,%i,%i,%i,%i,%i,%i"
			   )