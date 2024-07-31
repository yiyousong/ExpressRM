#nohup python test.py -b BAM/CRR072990_sorted.bam,BAM/CRR072998_sorted.bam,BAM/CRR073004_sorted.bam -o tmpout -s m6A_hg38_tissue_selected.rds &
import io
import numpy as np
import pandas as pd
import torch
import sys
import os
from torch import nn
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from functools import reduce
from subprocess import Popen
from Bio import SeqIO
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')
def fasta2binonehot(data):
    # data is a list of sequence: [n,seqlength]
    # possibly need list version where seqlength differ
    data=np.squeeze(np.array(list(map(list, data))))
    A = np.zeros_like(data,dtype=int)
    C = np.zeros_like(data,dtype=int)
    G = np.zeros_like(data,dtype=int)
    U = np.zeros_like(data,dtype=int)
    A[data == 'A'] = 1
    C[data == 'C'] = 1
    G[data == 'G'] = 1
    U[data == 'U'] = 1
    U[data == 'T'] = 1
    A = A[..., np.newaxis]
    C = C[..., np.newaxis]
    G = G[..., np.newaxis]
    U = U[..., np.newaxis]
    bindata=np.append(A,C,axis=-1)
    bindata = np.append(bindata, G, axis=-1)
    bindata = np.append(bindata, U, axis=-1)
    return bindata
class MultiAdaptPooling(nn.Module):
    def __init__(self, model, outsizelist=np.array([9, 25, 64])):
        super(MultiAdaptPooling, self).__init__()
        self.model = model
        self.modellist = []
        for i in outsizelist:
            self.modellist.append(nn.AdaptiveAvgPool1d(i))
    def forward(self, x):
        outlist = []
        for model in self.modellist:
            outlist.append(self.model(model(x)))
        out=torch.cat(outlist, -1)
        return out
dim = 64
droprate = 0.25
adaptoutsize = 15
geneinputsize = 28278
genelocinputsize = geneinputsize
class ExpressRM(pl.LightningModule):
    # unet assume seqlength to be ~500
    def __init__(self,useseq=True,usegeo=True,usetgeo=True,usegene=True,usegenelocexp=True, patchsize=7, patchstride=5, inchan=4, dim=64, kernelsize=7,
                 adaptoutsize=9, geneoutsize=500, geooutsize=32, droprate=0.25, lr=2e-5):
        super(ExpressRM, self).__init__()
        self.useseq = useseq
        self.usegeo = usegeo
        self.usegene = usegene
        self.usegenelocexp = usegenelocexp
        self.usetgeo = usetgeo
        self.droprate = droprate
        self.seqoutsize = 4 * adaptoutsize * dim
        self.geneoutsize = geneoutsize
        self.geooutsize = geooutsize
        self.learning_rate = lr
        self.posweight=torch.as_tensor(3.0)
        self.save_hyperparameters()
        self.conv_model = nn.Sequential(
            nn.Conv1d(in_channels=inchan, out_channels=dim, kernel_size=patchsize, stride=patchstride),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Dropout(droprate),
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernelsize),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Dropout(droprate),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernelsize),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Dropout(droprate))
        self.adaptconv_model = MultiAdaptPooling(
            nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernelsize),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(),
                nn.Dropout(droprate),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernelsize),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(),
                nn.Dropout(droprate),
                nn.AdaptiveAvgPool1d(adaptoutsize + 2*(kernelsize - 1)),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernelsize),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(),
                nn.Dropout(droprate),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernelsize),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(),
                nn.Dropout(droprate),
                nn.Flatten()
            )
            , np.array([16, 32, 64, 128]))
        self.geneenc = nn.Sequential(nn.Linear(28278, 1000), nn.LeakyReLU(), nn.Dropout(self.droprate),
                                     nn.Linear(1000, self.geneoutsize), nn.LeakyReLU())
        self.predicationhead = nn.Sequential(
            # nn.Flatten(1,-1),
            nn.Linear(self.seqoutsize + self.geneoutsize + 12 + 1, 2048),
            nn.LeakyReLU(),
            nn.Dropout(droprate),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(droprate),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(droprate),
            nn.Linear(1024, 4),
        )
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    def forward(self, x, geo, gene, genelocexp):
        batchsize = x.size()[0]
        tissuesize = genelocexp.size()[1]
        if self.useseq:
            x = x.transpose(-1, -2)
            adaptout = self.adaptconv_model(self.conv_model(x)).unsqueeze(-2).repeat(1,tissuesize,1)
        else:
            adaptout = torch.zeros([batchsize,tissuesize, self.seqoutsize]).float().cuda()
        # seq [N,2304]
        if self.usegene:
            # gene= self.geneenc(torch.mean(self.geneatt(geneloc,gene),dim=-2))
            gene= self.geneenc(gene).unsqueeze(0)
            gene=gene.repeat([1,batchsize//gene.size()[0],1,1]).flatten(0,1)
        else:
            gene= torch.zeros([batchsize,tissuesize,self.geneoutsize]).float().cuda()
            #[N,37,24]
        if not self.usetgeo:
                    geo[:,:,6:]*=0
        if not self.usegeo:
                geo[:, :, :6] *= 0
        if not self.usegenelocexp:
            genelocexp*=0
        adaptout = torch.cat([adaptout, gene, geo, genelocexp], dim=-1)
        out = self.predicationhead(adaptout)
        return out
file_path=os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('-b','--BAM', help='BAM is mandatory, each BAM file represent one transcriptome, seperate using "," ')
parser.add_argument('-o','--output', default=None, help='foldername for results(intermediates included),output will be prediction.csv and sitepredictionin this folder. Recommended to set manually for cases using more than one transcriptomes.  default remove 4 letters from BAMpath(xxx from xxx.bam)')
parser.add_argument('-s','--site', default='%s/m6A_hg38_tissue_selected.rds'%(file_path), help='site_path(grange.rds),default file contain non-site for evaluation, you may want to remove it to save time(~50%)')
parser.add_argument('-m','--model',default='%s/model.ckpt'%(file_path), help='model_path default same folder')
parser.add_argument('--refgene', default='%s/hg38.refGene.gtf'%(file_path), help='refgene.gtf location default same folder')
parser.add_argument('--knowngene', default='%s/hg38.knownGene.gtf'%(file_path), help='knowngene.gtf location default same folder')
parser.add_argument('--batchsize', default=1000, help='batchsize')
args, unknown = parser.parse_known_args()
if args.BAM is None:
    print('\n--------------------\n\nBAM is mandatory, each BAM file represent one transcriptome, seperate using "," \n\n--------------------\n')
    assert args.BAM is not None
BAM_list=args.BAM.split(',')
if args.output is None:
    args.output=BAM_list[0][:-4]
print('saving to folder:%s/'%(args.output))
os.system('mkdir -p %s'%(args.output))

commands =[]
for i in range(len(BAM_list)):
    BAM=BAM_list[i]
    commands.append('stringtie %s -e -G %s -A %s/geneexp%d.tab > %s/tmp.tmp'%(BAM,args.refgene,args.output,i,args.output))
    commands.append('stringtie %s -o %s/transcriptome%d.gtf > %s/tmp.tmp' % (BAM, args.output,i,args.output))
print('stringtie preprocessing this could take a long time')
procs = [ Popen(i, shell=True) for i in commands ]
for p in procs:
    p.wait()

print('calculating geographic encoding may take huge amounts of time(~ oneday for default 252009 sites)')
geneexpfile=''
for i in range(len(BAM_list)):
    geneexpfile+='%s/geneexp%d.tab,'%(args.output,i)
geneexpfile=geneexpfile[:-1]
Rcommands = ['Rscript gene.R %s %s %s %s %s'%(args.site,geneexpfile,args.output,args.refgene,file_path),
             'Rscript geo.R %s %s %s/geo.csv'%(args.site,args.knowngene,args.output)]
for i in range(len(BAM_list)):
    Rcommands.append(
             'Rscript geo.R %s %s/transcriptome%d.gtf %s/tgeo%d.csv'%(args.site,args.output,i,args.output,i))
procs = [ Popen(i, shell=True) for i in Rcommands ]
for p in procs:
   p.wait()
print('loading processed data')
seq_list=[]
for seq_record in SeqIO.parse('%s/sequence.fasta'%(args.output),format='fasta'):
    sequence=seq_record.seq
    seq_list.append(sequence)
seq_list=np.asarray(seq_list)
sequence=fasta2binonehot(seq_list)
print('sequence encoded')
geo_list=[]
for i in range(len(BAM_list)):
    refgeo=np.asarray(pd.read_csv('%s/geo.csv'%(args.output)))[:,6:]
    tgeo=np.asarray(pd.read_csv('%s/tgeo%d.csv'%(args.output,i)))[:,6:]
    geo=np.append(refgeo,tgeo,axis=-1)
    geo_list.append(geo)
geo=np.array(geo_list).transpose([1,0,2])
hostgeneexp=np.asarray(pd.read_csv('%s/lg2hosting_expression.csv'%(args.output)))
geneexp=np.asarray(pd.read_csv('%s/lg2geneexp.csv'%(args.output),index_col=0))
print('loading model')
model=ExpressRM().load_from_checkpoint(args.model,map_location=device)
model.eval()
sequence=torch.as_tensor(sequence).float().to(device)
geo=torch.as_tensor(geo).float().to(device)
hostgeneexp=torch.as_tensor(hostgeneexp).float().unsqueeze(2).to(device)
geneexp=torch.as_tensor(geneexp).float().transpose(1,0).to(device)
pred_list=[]
batchsize=int(args.batchsize)
for i in range(int(np.ceil(len(sequence)/batchsize))):
    pred=model.forward(sequence[batchsize*i:batchsize*(i+1)], geo[batchsize*i:batchsize*(i+1)], geneexp, hostgeneexp[batchsize*i:batchsize*(i+1)]).detach().cpu().numpy()
    pred_list.append(pred)
pred=np.concatenate(pred_list,axis=0)
np.savetxt('%s/prediction.csv'%(args.output),pred[...,0], delimiter=",")
np.savetxt('%s/siteprediction.csv'%(args.output),pred[...,3], delimiter=",")
