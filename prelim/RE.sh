#!/bin/bash
#
#
#$ -N glmer_test
#$ -q short.qc
#$ -pe shmem 8
#
### -cwd means work in the directory where submitted
#$ -cwd -V
#
### -j option combines output and error messages
#$ -j y
#$ -o output/glmer_48k.output
#$ -e output/glmer_48k.error

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

/usr/bin/time -v python /gpfs3/well/nichols/users/pra123/Longitudinal/prelim/RE_run.py

echo "*********************************"
echo "$JOB_ID finished at: `date`"
echo "*********************************"
exit 0
