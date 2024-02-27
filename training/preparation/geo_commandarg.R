library(rtracklayer)
library(GenomicRanges)
library(ensembldb)
library(Rsamtools)
library(matrixStats)
library(stats)
library("readxl")
library(stringr)
library(Homo.sapiens)
library(dplyr)
library(plyr)
gene_exp_list=c()
header_list=c()
### the terminal input should be the filename of the transcriptome. ref for reference genome ###
### looping is done outside the script for parallel processing ###
tissuename=commandArgs(trailingOnly = TRUE)


### must change the following path ###

### the following path is for:     ###
### work folder                    ###
### site in grange                 ###
### hg38 known genes in gtf format ###

mainfolder='/data1/yiyou/ExpressRM/'
seq2001=readRDS('/data1/yiyou/m6A_hg38_tissue_selected.rds')
refgtfpath='/data/bowen/ExpressRM/hg38.knownGene.gtf'

### change only if necessary ###
### following include the folder of the transcriptomes ###
### this script processes the gtffolder/tissuename.gtf
gtffolder=paste0(mainfolder,'/tissue/')
### or you can keep only the landmarkTX function and rewrite whole workflow ###
landmarkTX=function(seq2001,geogtf,matchtype='CDS'){
  geo=matrix(data=0,nrow=length(seq2001),ncol=12)
  geo[,1:6]=-1
  transidx=geogtf$type=='transcript'
  trans=geogtf[transidx]
  transidx=which(transidx==TRUE)
  ovlp=findOverlaps(seq2001,trans)
  for(idx in unique(ovlp@from)){
    mapidx=ovlp[ovlp@from==idx]@to
    if (length(mapidx)>1){
      maptrans=trans[mapidx]
      mapidx=mapidx[which(maptrans@ranges@width==max(maptrans@ranges@width))[1]]}
    maptrans=trans[mapidx]
    endidx=transidx[mapidx+1]-1
    if(is.na(endidx)){
      endidx=length(geogtf)
    }
    segs=geogtf[(transidx[mapidx]+1):endidx]
    if(matchtype=='CDS'){
    if (any(segs$type=='CDS')){
      CDS=segs[segs$type=='CDS']
    }else{
      CDS=segs[segs$type=='exon']
    }}else{CDS=segs[segs$type==matchtype]}
    exon=segs[segs$type=='exon']
    trans_dist_5=seq2001[idx]@ranges@start-maptrans@ranges@start
    trans_dist_3=maptrans@ranges@start+maptrans@ranges@width-1-seq2001[idx]@ranges@start
    CDS_dist_5=seq2001[idx]@ranges@start-min(CDS@ranges@start)
    CDS_dist_3=max(CDS@ranges@start+CDS@ranges@width-1)-seq2001[idx]@ranges@start
    exonovlp=findOverlaps(seq2001[idx],exon)
    if (identical(exonovlp@to, integer(0))){
      match_dist_5=-1
      match_dist_3=-1
    }else{
      match_dist_5=seq2001[idx]@ranges@start-exon[exonovlp@to]@ranges@start
      match_dist_3=exon[exonovlp@to]@ranges@start+exon[exonovlp@to]@ranges@width-seq2001[idx]@ranges@start-1
    }
    if (CDS_dist_5<0){CDS_dist_5=-1}
    if (CDS_dist_3<0){CDS_dist_3=-1}
    tmp=c(trans_dist_5,trans_dist_3,CDS_dist_5,CDS_dist_3,match_dist_5,match_dist_3)
    geo[idx,]=c(tmp,log2(tmp+2))
  }
  geo=data.frame(geo)
  colnames(geo)=c('trans_dist_5','trans_dist_3','CDS_dist_5','CDS_dist_3','match_dist_5','match_dist_3',
                  'logtrans_dist_5','logtrans_dist_3','logCDS_dist_5','logCDS_dist_3','logmatch_dist_5','logmatch_dist_3')
  return(geo)
}

if (tissuename=='ref'){
  knowngtf<- import(refgtfpath)
  geo=landmarkTX(seq2001,knowngtf)
  write.table(geo,paste0(mainfolder,'geo/geo.csv'),sep=',',row.names = FALSE)
}else{
  geogtf<- import(paste0(gtffolder,tissuename,'.gtf'))
  geo=landmarkTX(seq2001,geogtf,matchtype = 'exon')
  write.table(geo,paste0(mainfolder,'geo/',tissuename,'.csv'),sep=',',row.names = FALSE)
}

