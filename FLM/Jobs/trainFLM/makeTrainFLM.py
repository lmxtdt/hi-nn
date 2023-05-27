# -*- coding: utf-8 -*-

template = """#!/bin/bash -l
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=24g
#SBATCH --mail-type=END
#SBATCH --mail-user=thoms352@umn.edu
#SBATCH --output={abbr}{num}.out
#SBATCH --error={abbr}{num}.err

cd ~/DMI/FLM || exit
conda activate ML || exit

python3 PythonScripts/TrainFLM.py \\
        '{trainGlob}' '{valGlob}' '{neutralGlob}' \\
        {modelPath} {version} {load}  \\
        {kernel1} {kernel2} {kernel3} \\
        {filters1} {filters2} {filters3} \\
        {epochs} {batchSize} {learnRate} \\
        {plotPath}

conda deactivate
"""

abbr = "flmT"

num = 1 #

#fixed things

trainGlob = "SimData/trainNpz/*.npz"
valGlob = "SimData/valNpz/*.npz"
neutralGlob = "NeutData/npz/*.npz"

epochs = 25

learnRate = 1e-3
batchSize = 144

#version stuff
version = 0
if(version > 0):
    load = 1
else:
    load = 0

for version in range(1):
    if(version > 0):
        load = 1
    else:
        load = 0

    for kernel1 in [7, 11]:
        for kernel2 in [7, 11]:
            for kernel3 in [7, 11]:
                for filters1 in [32, 64]:
                    for filters2 in [32, 64]:
                        for filters3 in [32, 64]:
                            numFormatted = "{:03}".format(num)

                            #define model and plot paths
                            modelName = "model_k{:02}{:02}{:02}_f{:02}{:02}{:02}".format(
                                kernel1, kernel2, kernel3, filters1, filters2, filters3)
                            
                            modelPath = "Models/{}".format(modelName)
                            plotPath = "TrainPlots/v{}_{}.png".format(
                                version + 1,
                                modelName)

                            contents = template.format(abbr = abbr,
                                                       num = numFormatted,
                                                       modelPath = modelPath,
                                                       plotPath = plotPath,
                                                       version = version,
                                                       load = load,
                                                       trainGlob = trainGlob,
                                                       valGlob = valGlob,
                                                       neutralGlob = neutralGlob,
                                                       kernel1 = kernel1,
                                                       kernel2 = kernel2,
                                                       kernel3 = kernel3,
                                                       filters1 = filters1,
                                                       filters2 = filters2,
                                                       filters3 = filters3,
                                                       epochs = epochs,
                                                       batchSize = batchSize,
                                                       learnRate = learnRate)


                            with open("{}{}.sh".format(abbr, numFormatted), "w") as file:
                                    file.write(contents)

                            #advance
                            num += 1

print(num)
