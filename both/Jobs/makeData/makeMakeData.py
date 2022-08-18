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
    do for nMuts in {{1..12}}
        do ~/build/slim -d "numMuts = ${{nMuts}}" \\
                        -d "N = 1000" \\
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
    done;
done
conda deactivate
"""

abbr = "bothD"

jmax = 200
#12,000 simulations: ~1.4G individual ancestry data, compressed
#2.5G overall population ancestry data (compressed, 0.35G?)

num = 1 #51
saveIndData = "T"

for i in range(50):
    if(i == 10):
        saveIndData = "F"
    numFormatted = "{:02}".format(num)

    #various output paths
    simInfoOut = "Sims/sim/s_{}".format(numFormatted)
    ancPopOut = "Sims/pop/a_p_{}_".format(numFormatted)
    ancIndOut = "Sims/ind/a_i_{}_".format(numFormatted)
    fitOut = "Sims/fit/f_{}_".format(numFormatted)
    posOut = "Sims/pos/p_{}_".format(numFormatted)

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
