#!/bin/bash


#SBATCH -J beta0
#SBATCH -p short
#SBATCH --array=1-4
#
### -o option combines output and error messages
#SBATCH -o Mar23_output/beta0.output

echo "*********************************"
echo "Run on host: `hostname`"
echo "Operating system: `uname -s`"
echo "Username: `whoami`"
echo "Started at: `date`"
echo "*********************************"

# Setup
. /etc/profile
. ~/.bash_profile

module load R/3.6.2-foss-2019b

allBeta0=('-4' '-3' '-2' '-1')
arg=${allBeta0[$SLURM_ARRAY_TASK_ID-1]}
echo $arg

/usr/bin/time -v python /gpfs3/well/nichols/users/pra123/Longitudinal/Simulations/code/Mar23_rep_sims.py $arg 1.6 0.2 50 4 0.2 0.4 1000 3 /well/nichols/users/kindalov/FMRIB/Longitudinal/Simulations/Mar23_results/test_Beta0${SLURM_ARRAY_TASK_ID}.RData


echo "*********************************"
echo "$SLURM_JOBID finished at: `date`"
echo "*********************************"
exit 0
