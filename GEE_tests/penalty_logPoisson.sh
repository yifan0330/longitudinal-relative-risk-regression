#!/bin/bash
#
#
#$ -N penalty_geePK_logPoi
#$ -q short.qe
#$ -t 1-22
#
### -cwd means work in the directory where submitted
#$ -cwd -V
#
### -j option combines output and error messages
#$ -j y
#$ -o output/penalty_geePK_logPoi_exch25.output
#$ -e output/penalty_geePK_logPoi_exch25.error

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

/usr/bin/time -v python GEE_tests/GEE_penalty_logPoisson_run.py 1 $SGE_TASK_ID


echo "*********************************"
echo "$JOB_ID finished at: `date`"
echo "*********************************"
exit 0
