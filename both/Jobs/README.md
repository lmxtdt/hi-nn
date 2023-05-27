Jobs: all jobs.

---

makeData: runs SLiM simulations (with multDMI.slim). The ancestry files produced by SLiM are unnecessarily large, so after each simulation, ancestry data are compressed and the larger files are deleted. (Currently runs 120,000 simulations, 24,000 of which produce individual-level data, which amounts to around 17G of simulation output total.)

makeNeut: Uses the same process as makeData's file to run SLiM simulations, but for neutral simulations without any incompatibilities. It runs 20,000 simulations total, which all produce individual-level data.