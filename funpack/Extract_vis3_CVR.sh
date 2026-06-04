#!/bin/bash 
#
#$ -N vis3_CVR
#$ -P nichols.prjc
#$ -q short.qc
#$ -pe shmem 8
### -cwd means work in the directory where submitted
#$ -cwd -V
#


# -j option combines output and error messages
#$ -j y
#$ -o output/vis3_CVR.output
#$ -e output/vis3_CVR.error

# Setup
. /etc/profile
. ~/.bash_profile


SMS=$UKB_SMS
LongitudinalProj="/well/nichols/users/kindalov/FMRIB/Longitudinal/funpack"
 
funpack -v ${LongitudinalProj}/vars_CVR.txt \
	-vi 3 \
	${LongitudinalProj}/Vis3_CVR.tsv ${SMS}/ukb_latest.csv 

exit 0 
