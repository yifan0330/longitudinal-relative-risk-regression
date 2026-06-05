#!/bin/bash
#
#
#$ -N gee_rmd
#$ -q short.qe
#$ -t 1
#
### -cwd means work in the directory where submitted
#$ -cwd -V
#
### -j option combines output and error messages
#$ -j y
#$ -o output/geePK_rmd.output
#$ -e output/geePK_rmd.error

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

/usr/bin/time -v python GEE_tests/dispersion_check.py


echo "*********************************"
echo "$JOB_ID finished at: `date`"
echo "*********************************"
exit 0
