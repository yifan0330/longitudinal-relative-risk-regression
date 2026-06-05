#!/bin/bash
#
#
#$ -N beta0
#$ -q short.qe
#$ -t 1-3
#
### -cwd means work in the directory where submitted
#$ -cwd -V
#
### -j option combines output and error messages
#$ -j y
#$ -o Sept21_outputN50_phi/beta0.output
#$ -e Sept21_outputN50_phi/beta0.error

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

allBeta0=('-4' '-3' '-2')
arg=${allBeta0[$SGE_TASK_ID-1]}
echo $arg

/usr/bin/time -v python Simulations/code/rep_sims.py $arg 1.6 0.2 50 4 0.2 0.4 1000 3 Simulations/Sept21_resultsN50_phi/Beta0_$SGE_TASK_ID.RData


echo "*********************************"
echo "$JOB_ID finished at: `date`"
echo "*********************************"
exit 0
