#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=decode
#SBATCH --ntasks=32 --nodes=32
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.tang@yale.edu

module load Python/miniconda
# allennlp pykt
source activate allennlp

python ./runner.py
