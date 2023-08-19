#!/bin/bash

#SBATCH --partition=hard
#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=11000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

python main.py\
    tiff_path input\
    window_shape 16 16 16\
    --step 8\
    --clip_limit 0.01\
    --save_path 'output.tif'\
    --debug\
    
