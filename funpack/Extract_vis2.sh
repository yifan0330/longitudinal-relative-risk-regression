#!/bin/bash 
#
#$ -N vis2
#$ -P nichols.prjc
#$ -q short.qc
#$ -pe shmem 8
### -cwd means work in the directory where submitted
#$ -cwd -V
#


# -j option combines output and error messages
#$ -j y
#$ -o output/vis2.output
#$ -e output/vis2.error

# Setup
. /etc/profile
. ~/.bash_profile


SMS=$UKB_SMS
LongitudinalProj="/well/nichols/users/kindalov/FMRIB/Longitudinal/funpack"
 
funpack -v ${LongitudinalProj}/vars_vis23.txt \
	-vi 2 \
	${LongitudinalProj}/Vis2.tsv ${SMS}/ukb_latest.csv 

exit 0 
