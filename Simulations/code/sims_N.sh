#!/bin/bash
#
#
#$ -N Nsubj
#$ -q short.qe
#$ -t 1-6
#
### -cwd means work in the directory where submitted
#$ -cwd -V
#
### -j option combines output and error messages
#$ -j y
#$ -o Sept21_outputN50_phi/Nsubj.output
#$ -e Sept21_outputN50_phi/Nsubj.error

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
arg=${allN[$SGE_TASK_ID-1]}
echo $arg

/usr/bin/time -v python /gpfs3/well/nichols/users/pra123/Longitudinal/Simulations/code/rep_sims.py -4 1.6 0.2 $arg 4 0.2 0.4 1000 3 /well/nichols/users/kindalov/FMRIB/Longitudinal/Simulations/Sept21_resultsN50_phi/N${SGE_TASK_ID}.RData


echo "*********************************"
echo "$JOB_ID finished at: `date`"
echo "*********************************"
exit 0
