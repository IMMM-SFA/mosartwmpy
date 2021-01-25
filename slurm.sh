#!/usr/bin/env /bin/bash

#SBATCH -A im3
#SBATCH -N 1
#SBATCH -p short
#SBATCH -t 00:20:00
#SBATCH --exclusive
#SBATCH --job-name mosartwmpy

# TODO auto load correct modules, etc.

python launch.py
