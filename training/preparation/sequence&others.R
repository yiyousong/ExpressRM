library(GenomicRanges)
library(stringr)
library(plyr)
library(rtracklayer)
library(Biostrings)
#library(stats)
#library(dplyr)
#library(ensembldb)
#library(Homo.sapiens)
#library(Rsamtools)
#library(matrixStats)
#library("readxl")
gene_exp_list=c()
header_list=c()
parameters=commandArgs(trailingOnly = TRUE)
if (is.na(parameters[1])){mainfolder='/data1/yiyou/ExpressRM/'}else{mainfolder=parameters[1]}
if (is.na(parameters[2])){seqpath=paste0(mainfolder,'m6A_hg38_tissue_selected.rds')}else{seqpath=parameters[2]}
if (is.na(parameters[3])){refgtfpath=paste0(mainfolder,'hg38.refGene.gtf')}else{refgtfpath=parameters[3]}
if (is.na(parameters[4])){genetabfolder=paste0(mainfolder,'gene_expression_tab/')}else{genetabfolder=parameters[4]}
### must change the following path ###

### the following path is for:     ###
### work folder                    ###
### site in grange                 ###
### hg38 reference genes in gtf format ###
seq2001=readRDS(seqpath)
refg=import(refgtfpath)

### change only if necessary ###
genefolder=paste0(mainfolder,'gene_expression/')
filelist=list.files(genetabfolder)
tissuelist=str_sub(filelist,1,-5)
write(tissuelist,paste0(mainfolder,'tissuename.txt'))
transcripts=refg[refg$type=='transcript']
writeXStringSet(seq2001$refseq_2001,paste0(mainfolder,'selected.fasta'))

### comment out this section if the input file does not contain labels ###
label=data.frame(seq2001@elementMetadata[2:39])
colnames(label)=c('Site-label',colnames(seq2001@elementMetadata)[3:39])
label[1]=(label[1]=='P')
label=label*(data.frame(seq2001@elementMetadata[2])=='P')
label=label[ , c(order(colnames(label)[2:38])+1,1)]
write.table(label,paste0(mainfolder,'selectedlabel.csv'),sep=',',row.names = FALSE)
### comment out this section if the input file does not contain labels ###
### you still need to generate a selectedlabel.csv using other methods after removing this section ###

genetab_path=paste0(genetabfolder,tissuelist[1],'.tab')
gene_exp_table=read.table(genetab_path,header=1,sep='\t')
genename=sort(unique(gene_exp_table$Gene.ID))
genename2idx=data.frame(1:length(genename))
rownames(genename2idx)=genename
geneexp=matrix(0,nrow=length(genename),ncol = 38)
colnames(geneexp)=c('GeneName',tissuelist)
geneexp[,1]=genename
locexplist= data.frame(matrix(0,nrow = length(seq2001),ncol = 37))
colnames(locexplist)=tissuelist
for (i in 1:37){
  genetab_path=paste0(genetabfolder,tissuelist[i],'.tab')
  gene_exp_table=read.table(genetab_path,header=1,sep='\t')
  colnames(gene_exp_table)[3]='Seqnames'
  df=data.frame(gene_exp_table$Gene.ID,gene_exp_table$TPM)
  df=aggregate(gene_exp_table.TPM ~ gene_exp_table.Gene.ID, df, sum) 
  geneexp[genename2idx[df[,1],],i+1]=df[,2]
  gene_exp_grange=makeGRangesFromDataFrame(gene_exp_table,keep.extra.columns = TRUE)
  ovlp=findOverlaps(seq2001,gene_exp_grange)
  rowidx=ovlp@from
  repgene=gene_exp_grange[ovlp@to]
  locexp=data.frame('rowidx'=rowidx,'level'=repgene$TPM)
  locexp=aggregate(locexp$level ~ locexp$rowidx, locexp, sum) 
  locexplist[locexp[,1],i]=locexp[,2]
}
locexplist=log2(locexplist+1)
write.table(locexplist,paste0(genefolder,'lg2hosting_expression.csv'),sep=',',row.names = FALSE)
geneexp2=geneexp[,2:38]
geneexp2=as.double(geneexp2)
geneexp2=log2(geneexp2+1)
geneexp[,2:38]=geneexp2
write.table(geneexp,paste0(genefolder,'lg2geneexp.csv'),sep=',',col.names = TRUE,row.names = FALSE)