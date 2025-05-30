library(plyr)
library(ggplot2)
library(ggplot2)
library(cowplot)
library(patchwork)
library(stringr)                          
library("scales")         
library(stringr)

#stack all entry in one column 
path_list=list.files('./logs_1000/')
modellist=lapply(strsplit(path_list,'_'), `[[`, 3)
modellist=unlist(modellist)
modellist[modellist=='AdaptRM']='\nAdaptRM\n'
modellist[modellist=='seqgenegenelocexpgeotissuegeo']='\nExpressRM\n'
modellist=factor(modellist,levels=c("\nAdaptRM\n",
                                    '\nExpressRM\n',
                                    '\nExpressRM (untrained transcriptome)\n'))
sincolmat=data.frame(matrix(nrow = 0,ncol=6,dimnames = list(NULL,c('model','evaluation','testset','performance','tissueidx','not_trained'))))
evaluation_list=c('acc','auc','mcc')
for(k in 1:length(path_list)){
  metricpath=path_list[k]
  sitetestidx=unlist(strsplit(unlist(str_split(metricpath,'_'))[2],','))
  model=modellist[k]
  metric=read.table(paste0('/data1/yiyou/logs_1000/',metricpath,'/version_0/metrics.csv'),header = TRUE,sep=',')
  if (length(colnames(metric))>110){
    for (j in c(50,62,83)){
      idx=j:(j+2)
      tmp=metric[,idx]
      tmp=tmp[colSums(!is.na(tmp)) > 0]
      perf=na.omit(tmp)
      testset=rep(unlist(strsplit(colnames(perf)[1],'_'))[1],37)
      for(i in 1:3){
        sincolmat=rbind(sincolmat,data.frame(model=rep(model,37),evaluation=rep(evaluation_list[i],37),testset=testset,performance=unlist(perf[i]),tissueidx=1:37,not_trained=0:36 %in% sitetestidx))
      }}}
}
adaptperf=read.table(AdaptRM_path,header = TRUE,sep=',')
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
sincolmat[sincolmat$not_trained==TRUE,]$model='\nExpressRM (untrained transcriptome)\n'
selected=sincolmat


# visualize model performance after 1000 epochs
p=ggplot(selected[(selected$evaluation=='acc')&(selected$testset=='tissuetest'),],
        aes(x=model, y=performance,color=model,fill=model))+geom_violin(width=0.8,scale = "width",alpha=0.5)
g <- ggplot_build(p)
unlist(unique(g$data[[1]]["fill"]))
#must copy and re-define colors or graph will be black for unknown reason
colors11=c("#F8766D","#00BA38","#619CFF")
#colors11=c("#F8766D","#DB8E00","#AEA200","#64B200","#00BD5C","#00C1A7","#00BADE","#00A6FF","#B385FF","#EF67EB","#FF63B6")
# Identify hex codes
#,fill = after_scale(alpha('black', 0.1))
### plot ###
selected=sincolmat
#,fill = after_scale(alpha('black', 0.1))
p1 <- ggplot(selected[(selected$evaluation=='acc')&(selected$testset=='test'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p2 <- ggplot(selected[(selected$evaluation=='auc')&(selected$testset=='test'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p3 <- ggplot(selected[(selected$evaluation=='mcc')&(selected$testset=='test'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
leg <- get_legend(p1)
p1=p1+  ylab('Accuracy')+xlab('Modification prediction')+scale_x_discrete(position = "top")+
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(),legend.position = "none")
p2=p2+  ylab('AUROC')+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
                            axis.ticks.x=element_blank(),legend.position = "none")
p3=p3+  ylab('MCC')+ theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
                           axis.ticks.x=element_blank(),legend.position = "none")
p4 <- ggplot(selected[(selected$evaluation=='acc')&(selected$testset=='sitetest'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p5 <- ggplot(selected[(selected$evaluation=='auc')&(selected$testset=='sitetest'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p6 <- ggplot(selected[(selected$evaluation=='mcc')&(selected$testset=='sitetest'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p4=p4+xlab('Modifiable site prediction')+scale_x_discrete(position = "top")+
  theme(axis.text.x=element_blank(),axis.title.y=element_blank(),axis.ticks.x=element_blank(),legend.position = "none")
p5=p5+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.title.y=element_blank(),
            axis.ticks.x=element_blank(),legend.position = "none")
p6=p6+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.title.y=element_blank(),
            axis.ticks.x=element_blank(),legend.position = "none")
p7 <- ggplot(selected[(selected$evaluation=='acc')&(selected$testset=='tissuetest'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p8 <- ggplot(selected[(selected$evaluation=='auc')&(selected$testset=='tissuetest'),],
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p9 <- ggplot(selected[(selected$evaluation=='mcc')&(selected$testset=='tissuetest'),], 
             aes(x=model, y=performance,color=model,fill=model)) + geom_boxplot(width=0.3,fill=NA,alpha=1)+geom_violin(width=0.8,scale = "width",alpha=0.5)+ scale_fill_manual(values = colors11)
p7 =p7+xlab('Condition prediction')+scale_x_discrete(position = "top")+
  theme(axis.text.x=element_blank(),axis.title.y=element_blank(),axis.ticks.x=element_blank(),legend.position = "none")
p8 =p8+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.title.y=element_blank(),
            axis.ticks.x=element_blank(),legend.position = "none")
p9 =p9+theme(axis.title.x=element_blank(),axis.text.x=element_blank(),axis.title.y=element_blank(),
            axis.ticks.x=element_blank(),legend.position = "none")
p=(p1/p2/p3)|(p4/p5/p6)|(p7/p8/p9)|leg
p
ggsave('./epoch1000.pdf',p,width=12,height=8)              


# examine mean metrics of model performance
c(round(mean(sincolmat[(sincolmat$model=='\nExpressRM (untrained transcriptome)\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='test'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM (untrained transcriptome)\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='test'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM (untrained transcriptome)\n')&(sincolmat$evaluation=='mcc')&(sincolmat$testset=='test'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM (untrained transcriptome)\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='sitetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM (untrained transcriptome)\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='sitetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM (untrained transcriptome)\n')&(sincolmat$evaluation=='mcc')&(sincolmat$testset=='sitetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM (untrained transcriptome)\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='tissuetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM (untrained transcriptome)\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='tissuetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM (untrained transcriptome)\n')&(sincolmat$evaluation=='mcc')&(sincolmat$testset=='tissuetest'),]$performance),3)
)
c(round(mean(sincolmat[(sincolmat$model=='\nExpressRM\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='test'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='test'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM\n')&(sincolmat$evaluation=='mcc')&(sincolmat$testset=='test'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='sitetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='sitetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM\n')&(sincolmat$evaluation=='mcc')&(sincolmat$testset=='sitetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='tissuetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='tissuetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nExpressRM\n')&(sincolmat$evaluation=='mcc')&(sincolmat$testset=='tissuetest'),]$performance),3))

c(
  round(mean(sincolmat[(sincolmat$model=='\nAdaptRM\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='test'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nAdaptRM\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='test'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nAdaptRM\n')&(sincolmat$evaluation=='mcc')&(sincolmat$testset=='test'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nAdaptRM\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='sitetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nAdaptRM\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='sitetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nAdaptRM\n')&(sincolmat$evaluation=='mcc')&(sincolmat$testset=='sitetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nAdaptRM\n')&(sincolmat$evaluation=='acc')&(sincolmat$testset=='tissuetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nAdaptRM\n')&(sincolmat$evaluation=='auc')&(sincolmat$testset=='tissuetest'),]$performance),3),
  round(mean(sincolmat[(sincolmat$model=='\nAdaptRM\n')&(sincolmat$evaluation=='mcc')&(sincolmat$testset=='tissuetest'),]$performance),3))

