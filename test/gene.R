suppressMessages(library(GenomicRanges))
suppressMessages(library(stringr))
suppressMessages(library(plyr))
suppressMessages(library(rtracklayer))
suppressMessages(library(Biostrings))
suppressMessages(library(BSgenome.Hsapiens.UCSC.hg38))
#Rscript test.R BAM_path site_grange_path HG38_path 
tmp=commandArgs(trailingOnly = TRUE)
site_grange_path=tmp[1]
genetab_path=tmp[2]
out_folder=tmp[3]
reference_genome=tmp[4]
file_path=tmp[5]
seq2001=readRDS(site_grange_path)
genetab_list=unlist(strsplit(genetab_path,","))
gene_exp_table_list=list()
for(genetab in genetab_list){

  gene_exp_table=read.table(genetab,header=1,sep='\t')
  colnames(gene_exp_table)[3]='Seqnames'
  gene_exp_table_list=append(gene_exp_table_list,gene_exp_table)
}
library(BSgenome.Hsapiens.UCSC.hg38)
Hsapiens <- BSgenome.Hsapiens.UCSC.hg38
seq2=resize(seq2001,seq2001@ranges@width+2000,fix = 'center')
seq <- getSeq(Hsapiens,seq2)
writeXStringSet(seq,paste0(out_folder,'/sequence.fasta'))
genename=unlist(read.table(paste0(file_path,'/genename.csv')))
locexplist= data.frame(matrix(0,nrow = length(seq2001),ncol = length(genetab_list)))
genename2idx=data.frame(1:length(genename))
rownames(genename2idx)=genename
geneexp=matrix(0,nrow=length(genename),ncol = length(genetab_list)+1)
colnames(geneexp)=c('GeneName',c(1:length(genetab_list)))
geneexp[,1]=genename
for(i in 1:length(genetab_list)){
  genetab=genetab_list[i]
  gene_exp_table=read.table(genetab,header=1,sep='\t')
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
  locexplist[locexp[,1],i]=locexp[,2]}

  locexplist=log2(locexplist+1)
  write.table(locexplist,paste0(out_folder,'/lg2hosting_expression.csv'),sep=',',row.names = FALSE)
  geneexp2=geneexp[,2]
  geneexp2=as.double(geneexp2)
  geneexp2=log2(geneexp2+1)
  geneexp[,2]=geneexp2
  write.table(geneexp,paste0(out_folder,'/lg2geneexp.csv'),sep=',',col.names = TRUE,row.names = FALSE)