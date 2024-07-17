
library(ggplot2)
library(cowplot)
library(patchwork)
library(stringr)                          
library("scales")         
### generate a matrix containing all needed entries, but ggplot seems won't use use type of matrix ###
### df=matrix(data=0,nrow = 37,ncol=9)
### for (i in 14:22){
###   print(i)
###   df[,i-4]=perf[!is.na(perf[i]),i]
### }
### tmp=c()
### for (entry in colnames(perf)[14:22]){tmp=c(tmp,strsplit(entry,"_step")[[1]][1])}
### colnames(df)=tmp
############################

library(stringr)
### needs to stack all entry in one column ###
path_list=list.files('/data1/yiyou/lightning_logs/')
modellist=lapply(strsplit(path_list,'_'), `[[`, 3)
modellist=unlist(modellist)
modellist[modellist=='AdaptRM']=
  '\nAdaptRM\n'

modellist[modellist=='seq']='\nsequence\n'
modellist[modellist=='seqgene']='\nsequence\n+gene expression\n'
modellist[modellist=='seqgenegenelocexpgeotissuegeo']='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n'
modellist[modellist=='seqgenelocexpgeotissuegeo']='\nsequence\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n'
#modellist[modellist=='seqgenegenelocexptissuegeo']='\nsequence\n+gene expression\n+hosting expression\n+transcriptome geographic encoding\n'
modellist[modellist=='seqgenelocexp']='\nsequence\n+hosting expression\n'
modellist[modellist=='seqgenelocexptissuegeo']='\nsequence\n+hosting expression\n+transcriptome geographic encoding\n'
modellist[modellist=='seqgeo']='\nsequence\n+geographic encoding\n'
modellist[modellist=='seqtissuegeo']='\nsequence\n+transcriptome geographic encoding\n'
modellist[modellist=='seqgeotissuegeo']='\nsequence\n+geographic encoding\n+transcriptome geographic encoding\n'
modellist=factor(modellist,levels=c("\nAdaptRM\n",
                                    "\nsequence\n",
                                    "\nsequence\n+geographic encoding\n",
                                    "\nsequence\n+hosting expression\n",
                                    "\nsequence\n+gene expression\n",
                                    '\nsequence\n+transcriptome geographic encoding\n',
                                    '\nsequence\n+hosting expression\n+transcriptome geographic encoding\n',
                                    '\nsequence\n+geographic encoding\n+transcriptome geographic encoding\n',
                                    '\nsequence\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n',
                                    '\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n'))
sincolmat=data.frame(matrix(nrow = 0,ncol=6,dimnames = list(NULL,c('model','evaluation','testset','performance','tissueidx','not_trained'))))
evaluation_list=c('acc','auc','mcc')
for(k in 1:length(path_list)){
  metricpath=path_list[k]
  testidx=unlist(strsplit(unlist(str_split(metricpath,'_'))[2],','))
  model=modellist[k]
  metric=read.table(paste0('/data1/yiyou/lightning_logs/',metricpath,'/version_0/metrics.csv'),header = TRUE,sep=',')
  if (length(colnames(metric))>110){
  for (j in c(50,62,83)){
    idx=j:(j+2)
    tmp=metric[,idx]
    tmp=tmp[colSums(!is.na(tmp)) > 0]
    perf=na.omit(tmp)
    testset=rep(unlist(strsplit(colnames(perf)[1],'_'))[1],37)
    for(i in 1:3){
      sincolmat=rbind(sincolmat,data.frame(model=rep(model,37),evaluation=rep(evaluation_list[i],37),testset=testset,performance=unlist(perf[i]),tissueidx=1:37,not_trained=0:36 %in% testidx))
    }}}
    }
adaptperf=read.table('/data1/yiyou/lightning_logs/AdaptRM__AdaptRM_2001bp/version_0/metrics.csv',header = TRUE,sep=',')
for(j in c(15,18,21)){
  idx=j:(j+2)
  tmp=adaptperf[,idx]
  tmp=tmp[colSums(!is.na(tmp)) > 0]
  perf=na.omit(tmp)
  testset=rep(unlist(strsplit(colnames(perf)[1],'_'))[1],37)
  for(i in 1:3){
    newdata=data.frame(model=rep('\nAdaptRM\n',37),evaluation=rep(evaluation_list[i],37),testset=testset,performance=unlist(perf[i]),tissueidx=1:37,not_trained=rep(F,37))
    sincolmat=rbind(sincolmat,newdata)}}
sincolmat=ddply(sincolmat,c("model","testset",'tissueidx','evaluation','not_trained'),numcolwise(mean))
#selected=sincolmat[sincolmat$not_trained==F,]
#p=ggplot(selected[(selected$evaluation=='acc')&(selected$testset=='tissuetest'),], 
#         aes(x=model, y=performance,color=model,fill=model))+geom_violin(width=0.8,scale = "width",alpha=0.5)
#g <- ggplot_build(p)
#unlist(unique(g$data[[1]]["fill"]))
#must copy and re-define colors or graph will be black for unknown reason
#colors11=c("#F8766D","#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB", "#FF61C3")
#colors11=c("#F8766D","#D89000","#A3A500","#39B600","#00BF7D","#00BFC4","#00B0F6","#9590FF","#E76BF3","#FF62BC")
colors11=c("#F8766D","#DB8E00","#AEA200","#64B200","#00BD5C","#00C1A7","#00BADE","#00A6FF","#B385FF","#EF67EB","#FF63B6")
# Identify hex codes
#,fill = after_scale(alpha('black', 0.1))
### plot ###
selected=sincolmat[sincolmat$not_trained==F,]
#,fill = after_scale(alpha('black', 0.1))
p1 <- ggplot(selected[(selected$evaluation=='acc')&(selected$testset=='tissuetest'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p2 <- ggplot(selected[(selected$evaluation=='auc')&(selected$testset=='tissuetest'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p3 <- ggplot(selected[(selected$evaluation=='mcc')&(selected$testset=='tissuetest'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
leg <- get_legend(p1)
p1=p1+  ylab('Accuracy')+xlab('trained epitranscriptome')+scale_x_discrete(position = "top")+
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(),legend.position = "none")
p2=p2+  ylab('AUROC')+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
                            axis.ticks.x=element_blank(),legend.position = "none")
p3=p3+  ylab('MCC')+ theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
                           axis.ticks.x=element_blank(),legend.position = "none")
selected=sincolmat[sincolmat$not_trained==TRUE,]
#,fill = after_scale(alpha('black', 0.1))
p4 <- ggplot(selected[(selected$evaluation=='acc')&(selected$testset=='tissuetest'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+scale_x_discrete(position = "top",drop=F)+ scale_fill_manual(values = colors11[2:11])
p5 <- ggplot(selected[(selected$evaluation=='auc')&(selected$testset=='tissuetest'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+scale_x_discrete(drop=F)+ scale_fill_manual(values = colors11[2:11])
p6 <- ggplot(selected[(selected$evaluation=='mcc')&(selected$testset=='tissuetest'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+scale_x_discrete(drop=F)+ scale_fill_manual(values = colors11[2:11])
p4=p4+xlab('untrained epitranscriptome')+
  theme(axis.text.x=element_blank(),axis.title.y=element_blank(),axis.ticks.x=element_blank(),legend.position = "none")
p5=p5+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.title.y=element_blank(),
            axis.ticks.x=element_blank(),legend.position = "none")
p6=p6+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.title.y=element_blank(),
            axis.ticks.x=element_blank(),legend.position = "none")
p=(p1/p2/p3)|(p4/p5/p6)|leg
ggsave('tissuetest.pdf',p,width=10,height=8)              




selected=sincolmat[sincolmat$not_trained==F,]
#,fill = after_scale(alpha('black', 0.1))
p1 <- ggplot(selected[(selected$evaluation=='acc')&(selected$testset=='sitetest'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p2 <- ggplot(selected[(selected$evaluation=='auc')&(selected$testset=='sitetest'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p3 <- ggplot(selected[(selected$evaluation=='mcc')&(selected$testset=='sitetest'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
leg <- get_legend(p1)
p1=p1+  ylab('Accuracy')+xlab('trained epitranscriptome')+scale_x_discrete(position = "top")+
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(),legend.position = "none")
p2=p2+  ylab('AUROC')+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
                            axis.ticks.x=element_blank(),legend.position = "none")
p3=p3+  ylab('MCC')+ theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
                           axis.ticks.x=element_blank(),legend.position = "none")
selected=sincolmat[sincolmat$not_trained==TRUE,]
#,fill = after_scale(alpha('black', 0.1))
p4 <- ggplot(selected[(selected$evaluation=='acc')&(selected$testset=='sitetest'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+scale_x_discrete(position = "top",drop=F)+ scale_fill_manual(values = colors11[2:11])
p5 <- ggplot(selected[(selected$evaluation=='auc')&(selected$testset=='sitetest'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+scale_x_discrete(drop=F)+ scale_fill_manual(values = colors11[2:11])
p6 <- ggplot(selected[(selected$evaluation=='mcc')&(selected$testset=='sitetest'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+scale_x_discrete(drop=F)+ scale_fill_manual(values = colors11[2:11])
p4=p4+xlab('untrained epitranscriptome')+
  theme(axis.text.x=element_blank(),axis.title.y=element_blank(),axis.ticks.x=element_blank(),legend.position = "none")
p5=p5+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.title.y=element_blank(),
            axis.ticks.x=element_blank(),legend.position = "none")
p6=p6+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.title.y=element_blank(),
            axis.ticks.x=element_blank(),legend.position = "none")
p=(p1/p2/p3)|(p4/p5/p6)|leg
ggsave('sitetest.pdf',p,width=10,height=8)              






selected=sincolmat[sincolmat$not_trained==F,]
#,fill = after_scale(alpha('black', 0.1))
p1 <- ggplot(selected[(selected$evaluation=='acc')&(selected$testset=='test'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p2 <- ggplot(selected[(selected$evaluation=='auc')&(selected$testset=='test'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p3 <- ggplot(selected[(selected$evaluation=='mcc')&(selected$testset=='test'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
leg <- get_legend(p1)
p1=p1+  ylab('Accuracy')+xlab('trained epitranscriptome')+scale_x_discrete(position = "top")+
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(),legend.position = "none")
p2=p2+  ylab('AUROC')+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
                            axis.ticks.x=element_blank(),legend.position = "none")
p3=p3+  ylab('MCC')+ theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
                           axis.ticks.x=element_blank(),legend.position = "none")
selected=sincolmat[sincolmat$not_trained==TRUE,]
#,fill = after_scale(alpha('black', 0.1))
p4 <- ggplot(selected[(selected$evaluation=='acc')&(selected$testset=='test'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+scale_x_discrete(position = "top",drop=F)+ scale_fill_manual(values = colors11[2:11])
p5 <- ggplot(selected[(selected$evaluation=='auc')&(selected$testset=='test'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+scale_x_discrete(drop=F)+ scale_fill_manual(values = colors11[2:11])
p6 <- ggplot(selected[(selected$evaluation=='mcc')&(selected$testset=='test'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+scale_x_discrete(drop=F)+ scale_fill_manual(values = colors11[2:11])
p4=p4+xlab('untrained epitranscriptome')+
  theme(axis.text.x=element_blank(),axis.title.y=element_blank(),axis.ticks.x=element_blank(),legend.position = "none")
p5=p5+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.title.y=element_blank(),
            axis.ticks.x=element_blank(),legend.position = "none")
p6=p6+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.title.y=element_blank(),
            axis.ticks.x=element_blank(),legend.position = "none")
p=(p1/p2/p3)|(p4/p5/p6)|leg
ggsave('test.pdf',p,width=10,height=8)    


mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='tissuetest')&(!sincolmat$not_trained),]$performance)
mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='tissuetest')&(!sincolmat$not_trained),]$performance)
mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='tissuetest')&(sincolmat$not_trained),]$performance)
mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='tissuetest')&(sincolmat$not_trained),]$performance)
mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='test')&(!sincolmat$not_trained),]$performance)
mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='test')&(!sincolmat$not_trained),]$performance)
mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='test')&(sincolmat$not_trained),]$performance)
mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='test')&(sincolmat$not_trained),]$performance)
mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='sitetest')&(!sincolmat$not_trained),]$performance)
mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='sitetest')&(!sincolmat$not_trained),]$performance)
mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='sitetest')&(sincolmat$not_trained),]$performance)
mean(sincolmat[(sincolmat$model=='\nsequence\n+gene expression\n+hosting expression\n+geographic encoding\n+transcriptome geographic encoding\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='sitetest')&(sincolmat$not_trained),]$performance)