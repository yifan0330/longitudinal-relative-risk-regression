#!/bin/bash
#
#
#$ -N dispersed_geePK
#$ -q short.qe
#$ -t 1-22
#
### -cwd means work in the directory where submitted
#$ -cwd -V
#
### -j option combines output and error messages
#$ -j y
#$ -o output/geePK_disp.output
#$ -e output/geePK_disp.error

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

/usr/bin/time -v python /gpfs3/well/nichols/users/pra123/Longitudinal/GEE_tests/GEE_logPoisson_dispersed_run.py 1 $SGE_TASK_ID


echo "*********************************"
echo "$JOB_ID finished at: `date`"
echo "*********************************"
exit 0
