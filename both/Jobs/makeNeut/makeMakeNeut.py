# -*- coding: utf-8 -*-

templateStart = """#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=24g
#SBATCH --mail-type=END
#SBATCH --mail-user=thoms352@umn.edu
#SBATCH --output={abbr}{num}.out
#SBATCH --error={abbr}{num}.err

cd ~/DMI/both || exit
conda activate ML || exit

((rep = 0))

for j in {{1..{jmax}}}
    do ~/build/slim -d "numMuts = 1" \\
                    -d "N = 1000" \\
                    -d "neutral = 1" \\
                    -d "simInfoOutPath = '{simInfoOut}.csv'" \\
                    -d "ancPopOutPath = '{ancPopOut}${{rep}}.csv'" \\
                    -d "ancIndOutPath = '{ancIndOut}${{rep}}.txt'" \\
                    -d "saveIndData = {saveIndData}" \\
                    -d "fitOutPath = '{fitOut}${{rep}}.csv'" \\
                    -d "posOutPath = '{posOut}${{rep}}.csv'" \\
                    SlimScripts/multDMI.slim || exit
    python3 PythonScripts/CompressPop.py {ancPopOut}${{rep}}.csv {ancPopOut}${{rep}}.npz || exit
    rm {ancPopOut}${{rep}}.csv"""
templateCompressInd = """
    python3 PythonScripts/CompressInd.py {ancIndOut}${{rep}}.txt {ancIndOut}${{rep}}.npz || exit
    rm {ancIndOut}${{rep}}.txt"""
templateEnd = """
    ((rep += 1))
done
conda deactivate
"""

abbr = "bothD"

jmax = 2000

num = 1 #
saveIndData = "T"

for i in range(10):
    numFormatted = "{:02}".format(num)

    #various output paths
    simInfoOut = "NeutSims/sim/s_{}".format(numFormatted)
    ancPopOut = "NeutSims/pop/a_p_{}_".format(numFormatted)
    ancIndOut = "NeutSims/ind/a_i_{}_".format(numFormatted)
    fitOut = "NeutSims/fit/f_{}_".format(numFormatted)
    posOut = "NeutSims/pos/p_{}_".format(numFormatted)

    #write the file and save it
    contents = templateStart.format(abbr = abbr,
                           num = numFormatted,
                           jmax = jmax,
                           simInfoOut = simInfoOut,
                           ancPopOut = ancPopOut,
                           ancIndOut = ancIndOut,
                           saveIndData = saveIndData,
                           fitOut = fitOut,
                           posOut = posOut,
                           )
    if(saveIndData == "T"):
        contents += templateCompressInd.format(ancIndOut = ancIndOut)
    contents += templateEnd

    with open("{}{}.sh".format(abbr, numFormatted), "w") as file:
            file.write(contents)
            file.close()

    #advance
    num += 1


print(num)
