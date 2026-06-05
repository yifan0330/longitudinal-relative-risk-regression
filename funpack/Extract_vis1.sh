#!/bin/bash 
#
#$ -N vis1
#$ -P nichols.prjc
#$ -q short.qc
#$ -pe shmem 8
### -cwd means work in the directory where submitted
#$ -cwd -V
#


# -j option combines output and error messages
#$ -j y
#$ -o output/vis1.output
#$ -e output/vis1.error

# Setup
. /etc/profile
. ~/.bash_profile

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$PROJECT_ROOT"


SMS=$UKB_SMS
LongitudinalProj="funpack"
 
funpack -v 25734 \
    -s "v25734 != na" \
    -v ${LongitudinalProj}/vars_baseline.txt \
    -vi 1 \
    ${LongitudinalProj}/Vis1.tsv ${SMS}/ukb_latest.csv 

exit 0 
