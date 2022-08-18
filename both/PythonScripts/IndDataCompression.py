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
	values = []
	lengths = []
	splits = [0]
	
	#open file
	file = open(inTXT, "r")
	lines = []
	while(True):
		#read line from file
		line = file.readline().strip()
		lines.append(line)
		if(line == ''):
			#if the line ie empty, stop reading the file
			file.close()
			break
		else:
			#if the line is not empty, gather the values and when they change
			indValues = []
			indLengths = []
			
			currVal = 'X'
			lastBreak = 0
			#go through every character in the line
			for i in range(len(line)):
				#if it changes, record where it changes and to what
				if(line[i] != currVal):
					currVal = line[i]
					
					indValues.append(currVal)
					indLengths.append(i - lastBreak)
					
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
	
	return(ancArr)