All Python scripts specific to the FindLociModels.

---

ProcDataFLM.py: Processes the data from a group of simulations and writes to two output files: a .npz file containing the information loaded by DataGenFLM, i.e. X, Y, and meta identifying each chromosome; and a .csv containing additional meta information about each incompatibility in each simulation.

DataGenFLM.py: Defines data generators that load the .npz files produced by ProcDataFLM.py and provides easy access to their information.

ModelFLM.py: Defines the architecture of a FindLociModel, given various parameters.

TrainFLM.py: Trains a model. Can either create a new model using ModelFLM.py or load a pre-existing one. Uses DataGenFLM.py to define generators for the training and validation datasets.