#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=gen
#SBATCH --output=gen/gen-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=6000 
#SBATCH --time=5:00:00

context=$1
tr=$2

module load Python/miniconda
source activate allennlp

python ./generate.py $context $tr