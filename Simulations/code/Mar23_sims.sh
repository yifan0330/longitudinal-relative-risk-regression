#!/bin/bash


#SBATCH --job-name=alpha
#SBATCH -p short
#SBATCH --array=1-7
#
### -o option combines output and error messages
#SBATCH -o Mar23_output/alpha.output

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

allAlpha=('0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8')
arg=${allAlpha[$SLURM_ARRAY_TASK_ID-1]}
echo $arg

/usr/bin/time -v python /gpfs3/well/nichols/users/pra123/Longitudinal/Simulations/code/Mar23_rep_sims.py -4 1.6 0.2 50 4 0.2 $arg 1000 3 /well/nichols/users/kindalov/FMRIB/Longitudinal/Simulations/Mar23_results/test_alpha${SLURM_ARRAY_TASK_ID}.RData


echo "*********************************"
echo "$SLURM_JOBID finished at: `date`"
echo "*********************************"
exit 0
