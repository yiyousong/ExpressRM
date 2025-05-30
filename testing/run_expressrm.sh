#!/bin/bash
nohup python ./ExpressRM/testing/test.py -b ./data/input/CRR042278_sorted.bam, ./data/input/CRR073004_sorted.bam -o ./test -s ./data/input/m6A_hg38_tissue_selected.rds
