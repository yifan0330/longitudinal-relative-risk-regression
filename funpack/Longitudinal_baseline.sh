#!/bin/bash

SMS=$UKB_SMS

VisID=3 # 2nd visit is ID 3

cd $UKB_SUBJECTS/subjectsAll

ls -1d ${VisID}[0-9][0-9][0-9][0-9][0-9][0-9][0-9] | sed 's/^'"${VisID}"'//' > \
    $UKB_SMS/Subjsw2vis_a8107.txt

Out=${SMS}/ukb_latest-2ndVis

nice funpack -n -n -n \
    --overwrite \
    --num_jobs 12 \
    -s ${SMS}/Subjsw2vis_a8107.txt \
    --column eid34077 \
    ${Out}-bridge_a8107.tsv \
    ${SMS}/bridge_8107_34077.tsv

awk '(NR>1){print $2}' ${Out}-bridge_a8107.tsv > ${SMS}/Subjsw2vis_a34077.txt
