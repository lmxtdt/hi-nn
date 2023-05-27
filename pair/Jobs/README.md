procPair: processes simulation data (produced and held in both/) into .npz files that have been formatted and modified for the pair models. Each script (processing 2400 simulations) takes about 4 minutes to run.

trainPair: trains models over different parameters. (01 to 81 trains over layers1, layers2 = [1, 2, 4] and filters1, filters2 = [32, 64, 128])