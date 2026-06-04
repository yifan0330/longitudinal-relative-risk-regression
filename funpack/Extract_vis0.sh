#!/bin/bash 
#
#$ -N vis0
#$ -P nichols.prjc
#$ -q short.qc
#$ -pe shmem 8
### -cwd means work in the directory where submitted
#$ -cwd -V
#


# -j option combines output and error messages
#$ -j y
#$ -o output/vis0.output
#$ -e output/vis0.error

# Setup
. /etc/profile
. ~/.bash_profile


SMS=$UKB_SMS
LongitudinalProj="/well/nichols/users/kindalov/FMRIB/Longitudinal/funpack"
 
funpack -v 25734 \
    -s "v25734 != na" \
    -v ${LongitudinalProj}/vars_baseline.txt \
    -vi 0 \
    ${LongitudinalProj}/Vis0.tsv ${SMS}/ukb_latest.csv 

exit 0 
