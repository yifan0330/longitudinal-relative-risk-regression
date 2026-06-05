#!/bin/bash


#SBATCH -J betaB
#SBATCH -p short
#SBATCH --array 1-5
#
### -o option combines output and error messages
#SBATCH -o Mar23_output/betaB.output

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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

allBetaB=('1.2' '1.4' '1.6' '1.8' '2.0')
arg=${allBetaB[$SLURM_ARRAY_TASK_ID-1]}
echo $arg

/usr/bin/time -v python Simulations/code/Mar23_rep_sims.py -4 $arg 0.2 50 4 0.2 0.4 1000 3 Simulations/Mar23_results/test_BetaB_${SLURM_ARRAY_TASK_ID}.RData


echo "*********************************"
echo "$SLURM_JOBID finished at: `date`"
echo "*********************************"
exit 0
