#!/bin/bash
wget -P ./data/input/ http://www.rnamd.org/ExpressRM/m6A_hg38_tissue_selected.rds

nohup python ./ExpressRM/testing/test.py -b ./data/input/CRR072990_sorted.bam -o ./test -s ./data/input/m6A_hg38_tissue_selected.rds &
