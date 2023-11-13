suppressMessages(library(GenomicRanges))
suppressMessages(library(stringr))
suppressMessages(library(plyr))
suppressMessages(library(rtracklayer))
suppressMessages(library(Biostrings))
tmp=commandArgs(trailingOnly = TRUE)
site_grange_path=tmp[1]
gtf_path=tmp[2]
out_file=tmp[3]

landmarkTX=function(seq2001,geogtf,matchtype='exon'){
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
    exon=geogtf[(transidx[mapidx]+1):endidx]
    CDS=exon[exon$type==matchtype]
    trans_dist_5=seq2001[idx]@ranges@start-maptrans@ranges@start
    trans_dist_3=maptrans@ranges@start+maptrans@ranges@width-1-seq2001[idx]@ranges@start
    CDS_dist_5=seq2001[idx]@ranges@start-min(CDS@ranges@start)
    CDS_dist_3=max(CDS@ranges@start+CDS@ranges@width-1)-seq2001[idx]@ranges@start
    if (length(findOverlaps(seq2001[idx],CDS)@from)>0){
      match_dist_5=seq2001[idx]@ranges@start-CDS@ranges@start
      match_dist_5=min(match_dist_5[match_dist_5>=0])
      match_dist_3=(CDS@ranges@start+CDS@ranges@width)-seq2001[idx]@ranges@start-1
      match_dist_3=min(match_dist_3[match_dist_3>=0])}
    else{
      match_dist_5 = -1
      match_dist_3=-1
      if(CDS_dist_5<0){CDS_dist_5=-1}
      else{CDS_dist_3=-1}
    }
    tmp=c(trans_dist_5,trans_dist_3,CDS_dist_5,CDS_dist_3,match_dist_5,match_dist_3)
    stopifnot(tmp>-2)
    geo[idx,]=c(tmp,log2(tmp+2))
  }
  geo=data.frame(geo)
  colnames(geo)=c('trans_dist_5','trans_dist_3','CDS_dist_5','CDS_dist_3','match_dist_5','match_dist_3',
                  'logtrans_dist_5','logtrans_dist_3','logCDS_dist_5','logCDS_dist_3','logmatch_dist_5','logmatch_dist_3')
  return(geo)
}
seq2001=readRDS(site_grange_path)
geogtf= import(gtf_path)
geo=landmarkTX(seq2001,geogtf,matchtype = 'exon')
write.table(geo,out_file,sep=',',row.names = FALSE)