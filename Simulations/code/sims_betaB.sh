#!/bin/bash
#
#
#$ -N betaB
#$ -q short.qe
#$ -t 1-5
#
### -cwd means work in the directory where submitted
#$ -cwd -V
#
### -j option combines output and error messages
#$ -j y
#$ -o Sept21_outputN50_phi/betaB.output
#$ -e Sept21_outputN50_phi/betaB.error

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

allBetaB=('1.2' '1.4' '1.6' '1.8' '2.0')
arg=${allBetaB[$SGE_TASK_ID-1]}
echo $arg

/usr/bin/time -v python /gpfs3/well/nichols/users/pra123/Longitudinal/Simulations/code/rep_sims.py -4 $arg 0.2 50 4 0.2 0.4 1000 3 /well/nichols/users/kindalov/FMRIB/Longitudinal/Simulations/Sept21_resultsN50_phi/BetaB_$SGE_TASK_ID.RData


echo "*********************************"
echo "$JOB_ID finished at: `date`"
echo "*********************************"
exit 0
