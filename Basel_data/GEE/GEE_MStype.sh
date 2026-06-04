#!/bin/bash
#
#
#$ -N gee_MS
#$ -q short.qc
#$ -pe shmem 8
#
### -cwd means work in the directory where submitted
#$ -cwd -V
#
### -j option combines output and error messages
#$ -j y
#$ -o output/geepack_MS_93k.output
#$ -e output/geepack_MS_93k.error

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

/usr/bin/time -v python /gpfs3/well/nichols/users/pra123/Longitudinal/Basel_data/GEE/GEE_MStype.py

echo "*********************************"
echo "$JOB_ID finished at: `date`"
echo "*********************************"
exit 0
