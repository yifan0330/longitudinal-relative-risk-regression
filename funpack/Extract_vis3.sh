#!/bin/bash 
#
#$ -N vis3
#$ -P nichols.prjc
#$ -q short.qc
#$ -pe shmem 8
### -cwd means work in the directory where submitted
#$ -cwd -V
#


# -j option combines output and error messages
#$ -j y
#$ -o output/vis3.output
#$ -e output/vis3.error

# Setup
. /etc/profile
. ~/.bash_profile

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$PROJECT_ROOT"


SMS=$UKB_SMS
LongitudinalProj="funpack"
 
funpack -v ${LongitudinalProj}/vars_vis23.txt \
    -vi 3 \
    ${LongitudinalProj}/Vis3.tsv ${SMS}/ukb_latest.csv 

exit 0 
