import os
import sys
import time
import argparse
from subprocess import Popen
parser = argparse.ArgumentParser()
parser.add_argument('--mainfolder',help='folder to dump everything (with / at end)')
parser.add_argument('--gtffolder', default=None,help='folder containing transcriptome genomes (WITHOUT reference genomes)')
parser.add_argument('--genetabfolder', default=None,help='folder containing gene expression in .tab format')
parser.add_argument('--seqpath', default=None,help='m6A_hg38_tissue_selected.rds')
parser.add_argument('--refgtfpath', default=None,help='hg38.refGene.gtf')
parser.add_argument('--knowngtfpath', default=None,help='hg38.knownGene.gtf')
args, unknown = parser.parse_known_args()
if args.gtffolder is None:
    args.gtffolder=args.mainfolder+'transcriptomes/'
if args.genetabfolder is None:
    args.genetabfolder=args.mainfolder+'gene_expression_raw/'
if args.seqpath is None:
    args.seqpath=args.mainfolder+'m6A_hg38_tissue_selected.rds'
if args.refgtfpath is None:
    args.refgtfpath=args.mainfolder+'hg38.refGene.gtf'
if args.knowngtfpath is None:
    args.knowngtfpath=args.mainfolder+'hg38.knownGene.gtf'

os.system('nohup Rscript sequence&others.R %s %s %s %s &'%(args.mainfolder,args.seqpath,args.refgtfpath,args.genetabfolder))

### looping is done outside the Rscript for parallel processing ###
commands =[]
commands.append('nohup Rscript geo_commandarg.R ref %s %s %s %s &' % ( args.mainfolder, args.seqpath, args.knowngtfpath, args.gtffolder))
for file in os.listdir(gtffolder):
    commands.append('nohup Rscript geo_commandarg.R %s %s %s %s %s &'%(file[:-4],args.mainfolder,args.seqpath,args.knowngtfpath,args.gtffolder))
procs = [ Popen(i, shell=True) for i in commands ]
for p in procs:
    p.wait()
os.system('nohup python to_tensor.py %s &'%(args.mainfolder))
