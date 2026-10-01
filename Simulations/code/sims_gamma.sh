#!/bin/bash
#
#
#$ -N gammaN500
#$ -q short.qe
#$ -t 1-7
#
### -cwd means work in the directory where submitted
#$ -cwd -V
#
### -j option combines output and error messages
#$ -j y
#$ -o Sept21_N500_outputN50_phi/gamma.output
#$ -e Sept21_N500_outputN50_phi/gamma.error

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

allGamma=('0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8')
arg=${allGamma[$SGE_TASK_ID-1]}
echo $arg

/usr/bin/time -v python Simulations/code/rep_sims.py -4 1.6 0.2 500 4 $arg 0.4 1000 3 Simulations/Sept21_resultsN50_phi/N500_gamma$SGE_TASK_ID.RData


echo "*********************************"
echo "$JOB_ID finished at: `date`"
echo "*********************************"
exit 0
