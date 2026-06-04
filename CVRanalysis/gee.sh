#!/bin/bash
#
#
#$ -N CVR_July_geePK
#$ -q short.qe
#$ -t 1-40
#
### -cwd means work in the directory where submitted
#$ -cwd -V
#
### -j option combines output and error messages
#$ -j y
#$ -o output/RRgee_July_Ageinteraction_CVR.output
#$ -e output/RRgee_July_Ageinteraction_CVR.error

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

/usr/bin/time -v python /gpfs3/well/nichols/users/pra123/Longitudinal/CVRanalysis/code/GEE_logPoisson_interaction_run.py 1 $SGE_TASK_ID


echo "*********************************"
echo "$JOB_ID finished at: `date`"
echo "*********************************"
exit 0
