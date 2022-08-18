#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compress population data (or any int array saved as a csv file) into a
.npz file, saving the array under the keyword "pop"

Created August 18, 2022
Last edited August 18, 2022
"""

import numpy as np
from csv import reader
from sys import argv

inFile = argv[1]
outFile = argv[2]

def compress(inFile, outFile):
    rows = []
    with open(inFile, "r") as file:
        r = reader(file)
        for row in r:
            rows.append(row)
            
    arr = np.array(rows, dtype = np.int32)

    np.savez_compressed(outFile,
                        pop = arr)
    
compress(inFile, outFile)