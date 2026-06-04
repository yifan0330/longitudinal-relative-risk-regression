#!/bin/bash


#SBATCH -J N
#SBATCH -p short
#SBATCH --array=1-6
#
### -o option combines output and error messages
#SBATCH -o Mar23_output/N.output

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

allN=('25' '50' '75' '100' '500' '1000')
arg=${allN[$SLURM_ARRAY_TASK_ID-1]}
echo $arg

/usr/bin/time -v python /gpfs3/well/nichols/users/pra123/Longitudinal/Simulations/code/Mar23_rep_sims.py -4 1.6 0.2 $arg 4 0.2 0.4 1000 3 /well/nichols/users/kindalov/FMRIB/Longitudinal/Simulations/Mar23_results/test_N_${SLURM_ARRAY_TASK_ID}.RData


echo "*********************************"
echo "$SLURM_JOBID finished at: `date`"
echo "*********************************"
exit 0
