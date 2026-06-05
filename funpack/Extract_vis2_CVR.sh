#!/bin/bash 
#
#$ -N vis2_CVR
#$ -P nichols.prjc
#$ -q short.qc
#$ -pe shmem 8
### -cwd means work in the directory where submitted
#$ -cwd -V
#


# -j option combines output and error messages
#$ -j y
#$ -o output/vis2_CVR.output
#$ -e output/vis2_CVR.error

# Setup
. /etc/profile
. ~/.bash_profile

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$PROJECT_ROOT"


SMS=$UKB_SMS
LongitudinalProj="funpack"
 
funpack -v ${LongitudinalProj}/vars_CVR.txt \
	-vi 2 \
	${LongitudinalProj}/Vis2_CVR.tsv ${SMS}/ukb_latest.csv 

exit 0 
