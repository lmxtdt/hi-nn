Jobs, and the Python scripts that generate them, regarding FindLociModel.

---

procFLM: processes simulation data (produced and stored in the both folder) into useable training (or validation or testing) data. Each script (processing 2400 simulations) takes around 30 minutes to run.

procNeut: same as procFLM, but it processes neutral simulation data.

trainFLM: trains models over different parameters. (001 to 064 trains over filters1, filters2, filters3 = [32, 64] and kernel1, kernel2 = [7, 19], with batchSize = 144 and learnRate = 1e-3.)