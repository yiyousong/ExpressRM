peaks=readRDS('/home/share/bowen/86backup/ExpressRM/ExpressRM/tissue/GOS/peak.rds')
prediction=read.csv('/home/share/yiyou.song/test/out/geo.csv',header=FALSE)
site_grange_path='/home/share/yiyou.song/test/m6A_hg38_tissue_selected.rds'
seq2001=readRDS(site_grange_path)
ovlp=findOverlaps(seq2001,peaks)
write.table(ovlp@from,'label.csv',col.names = FALSE,row.names = FALSE,sep = ',')
