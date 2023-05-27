# -*- coding: utf-8 -*-

template = """#!/bin/bash -l
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=24g
#SBATCH --mail-type=END
#SBATCH --mail-user=thoms352@umn.edu
#SBATCH --output={abbr}{num}.out
#SBATCH --error={abbr}{num}.err

cd ~/DMI/pair || exit
conda activate ML || exit

python3 PythonScripts/TrainPair.py {modelPath} {version} {load} \\
        '{trainGlob}' '{valGlob}' \\
        {layers1} {layers2} {filters1} {filters2} \\
        {distEncode} {dEarly} \\
        {epochs} {batchSize} {learnRate} \\
        {plotPath}

conda deactivate
"""

abbr = "pairT"

num = 1 #

#fixed things

trainGlob = "SimData/trainNpz/*.npz"
valGlob = "SimData/valNpz/*.npz"

distEncode = "inc10"
dEarly = 0

epochs = 50
batchSize = 144
learnRate = 1e-4

#version stuff
version = 0
if(version > 0):
    load = 1
else:
    load = 0


for layers1 in [1, 2, 4]:
    for layers2 in [1, 2, 4]:
        for filters1 in [32, 64, 128]:
            for filters2 in [32, 64, 128]:
                numFormatted = "{:02}".format(num)

                #define model and plot paths
                modelPath = "Models/model_l{}_l{}_f{}_f{}".format(
                    layers1, layers2, filters1, filters2)
                plotPath = "TrainPlots/v{}_model_l{}_l{}_f{}_f{}.png".format(
                    version + 1,
                    layers1, layers2, filters1, filters2)

                contents = template.format(abbr = abbr,
                                           num = numFormatted,
                                           modelPath = modelPath,
                                           plotPath = plotPath,
                                           version = version,
                                           load = load,
                                           trainGlob = trainGlob,
                                           valGlob = valGlob,
                                           layers1 = layers1,
                                           layers2 = layers2,
                                           filters1 = filters1,
                                           filters2 = filters2,
                                           distEncode = distEncode,
                                           dEarly = dEarly,
                                           epochs = epochs,
                                           batchSize = batchSize,
                                           learnRate = learnRate)


                with open("{}{}.sh".format(abbr, numFormatted), "w") as file:
                        file.write(contents)

                #advance
                num += 1

print(num)
