Python scripts relevant to simulations and compressing their data.

---

IndDataCompression.py: Contains functions for compressing and decompressing the .txt files containing individual ancestry data which are produced by the SLiM scripts.

CompressInd.py: Compresses individual-level ancestry (simply calls compress from IndDataCompression.py).

CompressPop.py: Compresses population-level ancestry, by producing a compressed .npz file from the original .csv file.