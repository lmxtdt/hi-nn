Python scripts relevant to the PairsModels.

---

ProcDataPair.py: Processes simulation data (which was produced and held in both/) into a format relevant to the PairsModels. For each simulation, it extracts all true incompatibility pairs and generates an equal number of fake pairs (two loci not in an incompatibility together). Per-individual ancestry at both loci is saved. It is capable of both using all surviving individuals from a population, or using exactly 500 individuals (which is necessary for training).

DataGenPair.py: Defines data generators that load the data saved by ProcDataPair.py in a convenient way. Used for training and evaluation.

ModelPair.py: Defines the TensorFlow model used for the PairsModels. Can create a new model, given the necessary parameters, or load a pre-existing one. (Note: technically, it creates two TensorFlow models, one which is used for training and has a fixed input size, and one which can be used to run predictions on samples with a variable number of individuals.)

TrainPair.py: Trains a PairsModel. Can create a new model or load a pre-existing one. Feeds data into the model using a generator from DataGenPair.py.