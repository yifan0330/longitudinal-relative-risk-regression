#!/bin/bash
#
#
#$ -N interaction_July_pgeePK
#$ -q short.qe
#$ -t 1-56
#
### -cwd means work in the directory where submitted
#$ -cwd -V
#
### -j option combines output and error messages
#$ -j y
#$ -o output/pgeePK_July_interaction.output
#$ -e output/pgeePK_July_interaction.error

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

/usr/bin/time -v python Apr2021_GEE/code/PGEE_logPoisson_interaction_run.py 1 $SGE_TASK_ID


echo "*********************************"
echo "$JOB_ID finished at: `date`"
echo "*********************************"
exit 0
