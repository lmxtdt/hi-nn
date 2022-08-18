# -*- coding: utf-8 -*-

template = """#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=24g
#SBATCH --mail-type=END  
#SBATCH --mail-user=thoms352@umn.edu
#SBATCH --output={abbr}{num}.out
#SBATCH --error={abbr}{num}.err

cd ~/DMI/both || exit

conda activate ML || exit

python3 ~/DMI/FLM/PythonScripts/ProcDataFLM.py {inCSV} {outNPZ} {outMeta} || exit

conda deactivate

"""

abbr = "flmPr"

num = 1 #

for i in range(50):
    numFormatted = "{:02}".format(num)

    #various output paths
    simInfo = "Sims/sim/s_{}.csv".format(numFormatted)
    npzOut = "../FLM/SimData/npz/n_{}.npz".format(numFormatted)
    metaOut = "../FLM/SimData/meta/m_{}.csv".format(numFormatted)


    #write the file and save it
    contents = template.format(abbr = abbr,
                           num = numFormatted,
                           inCSV = simInfo,
                           outNPZ = npzOut,
                           outMeta = metaOut
                           )

    with open("{}{}.sh".format(abbr, numFormatted), "w") as file:
            file.write(contents)

    #advance
    num += 1


print(num)
