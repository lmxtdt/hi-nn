#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndDataCompression

Functions for a) compressing individual-level data from the SliM .txt files
and b) expanding the compressed format into a full array 

Created August 5, 2022
Last edited August 5, 2022
"""
import numpy as np

def compress(inTXT, outNPZ):
	"""
	Compress individual-level ancestry stored in a .txt file.

	Parameters
	----------
	inTXT : string
		Path to the individual ancestry .txt file to use as input.
	outNPZ : string
		Path for the output, which will be written as a compressed .npz file.
	"""
	
	#Because an individual's ancestry is the same over contiguous swathes
	#of their genome, an individual's ancestry is compressed to the length of
	#those contiguous regions and the corresponding values for the regions.
	
	#initialize lists
	values = []  #the ancestry for the regions, for all individuals
	lengths = [] #the lengths of the regions, for all individuals
	splits = [0] #splits to be used in separating individuals from each other
	
	#open file
	file = open(inTXT, "r")
	lines = [] #list of all the lines in the file
	#read each line in the file
	while(True):
		#read line from file
		line = file.readline().strip()
		lines.append(line)
		
		if(line == ''):
			#if the line is empty, stop reading the file
			file.close()
			break
		else:
			#if the line is not empty, gather the values and when they change
			indValues = [] #ancestry for the regions of this individual
			indLengths = [] #lengths of the regions of this individual
			
			currVal = 'X'
			lastBreak = 0
			#go through every character in the line
			for i in range(len(line)):
				#if it changes, record where it changes and to what
				if(line[i] != currVal):
					#update currVal
					currVal = line[i]
					
					#add information about the values and lengths
					indValues.append(currVal)
					indLengths.append(i - lastBreak)
					
					#update lastBreak
					lastBreak = i
			indLengths.append(1 + i - lastBreak)

			#save the individual's ancestry to the overall lists
			values.append(indValues)
			lengths.append(indLengths[1:])
			splits.append(splits[-1] + len(indValues))
	#cut the first and last values off splits, they are unnecessary
	splits = splits[1:-1]
			
	#save the file
	np.savez_compressed(outNPZ,
		  values = np.concatenate(values).astype(np.uint8),
		  lengths = np.concatenate(lengths),
		  splits = splits)

def decompress(inNPZ, convertToFloat = True):
	"""
	Read compressed individual-level ancestry stored in a .npz file and return
	the longform ancestry array.

	Parameters
	----------
	inNPZ : string
		Path to the compressed .npz file.
	convertToFloat : bool, optional
		Whether to convert the ancestry to floats (0, 0.5, 1). If False, 
		ancestry stays as ints (0, 1, 2). The default is True.

	Returns
	-------
	Numpy array of shape (number individuals, number loci) representing the
	ancestry of each individual over all loci.
	"""
	#read file
	npzFile = np.load(inNPZ)
	allValues = npzFile.get("values")
	allLengths = npzFile.get("lengths")
	splits = npzFile.get("splits")
	npzFile.close()
	
	#split values and lengths to be per individual
	values = np.split(allValues, splits)
	lengths = np.split(allLengths, splits)
	
	#for each individual, get their ancestry
	anc = []
	for i in range(len(values)):
		anc.append(np.repeat(values[i], lengths[i]))
	
	ancArr = np.stack(anc)
	if(convertToFloat):
		ancArr = ancArr.astype(float)
		ancArr[ancArr == 1] = 0.5
		ancArr[ancArr == 2] = 1
	
	return ancArr