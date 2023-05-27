# -*- coding: utf-8 -*-

template = """#!/bin/bash -l
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=24g
#SBATCH --mail-type=END  
#SBATCH --mail-user=thoms352@umn.edu
#SBATCH --output={abbr}{num}.out
#SBATCH --error={abbr}{num}.err

cd ~/DMI/both || exit

conda activate ML || exit

python3 ~/DMI/pair/PythonScripts/ProcDataPair.py {inCSV} {outNPZ} {ragged} || exit

conda deactivate

"""

abbr = "pairPr"

num = 1 #

ragged = 0

for i in range(50):
    numFormatted = "{:02}".format(num)

    #various output paths
    inCSV = "Sims/sim/s_{}.csv".format(numFormatted)
    outNPZ = "../pair/SimData/npz/n_{}.npz".format(numFormatted)

    #write the file and save it
    contents = template.format(abbr = abbr,
                           num = numFormatted,
                           inCSV = inCSV,
                           outNPZ = outNPZ,
                           ragged = ragged,
                           )

    with open("{}{}.sh".format(abbr, numFormatted), "w") as file:
            file.write(contents)

    #advance
    num += 1


print(num)
