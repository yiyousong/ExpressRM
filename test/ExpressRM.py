# folder_prefix = '/gpfs/work/bio/jiameng/yiyou/V3'
folder_prefix = '/gpfs/work/bio/yiyousong15/ExpressRMV1.1'
# data_location = '/gpfs/work/bio/yiyousong15/ExpressRMv3/data/'
genevaepath='/gpfs/work/bio/yiyousong15/ExpressRM/model/BTCGeneVAE/epoch=61-step=6200.ckpt'
# folder_prefix = '/data1/yiyou/ExpressRMv2'
# data_location = '/data1/yiyou/ExpressRMv2/data/'
data_location='/gpfs/work/bio/yiyousong15/ExpressRMV1.1/tensor/'
dim = 64
droprate = 0.25
adaptoutsize = 15
geneinputsize = 28278
genelocinputsize = geneinputsize
from Layers import *
import io
import numpy as np
import pandas as pd
import torch
import sys
import os
from torch import nn
import argparse
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
import torchmetrics.classification as C
import gc
from functools import reduce
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
# torch.set_num_threads(10)
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None, help='add path to continue training else start anew')
parser.add_argument('--seq', default=True, help='')
parser.add_argument('--gene', default=False, help='')
parser.add_argument('--genelocexp', default=False, help='')
parser.add_argument('--geo', default=False, help='')
parser.add_argument('--tgeo', default=False, help='')
parser.add_argument('--featurelist', default=None, help='1,0,0,0,0\n sequence,gene,genelocexp,geo,tgeo')
parser.add_argument('--radius', default=1000, help='2*radius+1')
parser.add_argument('--epoch', default=0, help='')
parser.add_argument('--prefix', default='', help='')
parser.add_argument('--autoprefix', default=True, help='')
parser.add_argument('--trainlist', default=None, help='use testlist')
parser.add_argument('--testlist', default=None, help='drop idx (during training) or leaveoneout')
parser.add_argument('--precision', default=32, help='precision')
args, unknown = parser.parse_known_args()
if args.featurelist is None:
    useseq = bool(int(args.seq))
    usegene = bool(int(args.gene))
    usegenelocexp = bool(int(args.genelocexp))
    usegeo = bool(int(args.geo))
    usetgeo = bool(int(args.tgeo))
else:
    useseq,usegene,usegenelocexp,usegeo,usetgeo=np.asarray(args.featurelist.split(',')).astype(bool)
radius = int(args.radius)
prefix = args.prefix
if args.trainlist is None:
    train_list = np.arange(37)
else:
    train_list = list(map(int, args.trainlist.split(',')))
if args.testlist is not None:
        prefix += 'testidx_'
        prefix += args.testlist
        test_list = []
        for testidx in args.testlist.split(','):
            test_list.append(int(testidx))
        train_list = np.delete(train_list, test_list)
if args.autoprefix != 'false':
    prefix += '_'
    if useseq:
        prefix += 'seq'
    if usegene:
        prefix += 'gene'
    if usegenelocexp:
        prefix += 'genelocexp'
    if usegeo:
        prefix += 'geo'
    if usetgeo:
        prefix += 'tissuegeo'
in_chan = 4
prefix += '_%dbp' % (2 * radius + 1)
def weightlabel(labelwithsite):
    label=labelwithsite[...,:-1]
    sitelabel = labelwithsite[...,-1:].repeat(1, label.shape[1])
    weight = 1.1 * label / (torch.sum(label, axis=1).unsqueeze(-1) + 1e-4) + (1 - label) / (
                torch.sum(1 - label, axis=1).unsqueeze(-1) + 1e-4)
    weight*=label.shape[1]/2
    try:
        assert torch.sum(weight < 0) == 0
    except:
        print(weight)
        print(label)
        assert all(weight >= 0)
    return label,sitelabel,weight
class trainDataset(Dataset):
    def __init__(self, dataidx=None,tissueidx=np.arange(37),radius=1000):
        self.radius=radius
        self.gene=torch.load('%slg2geneexp.pt'%(data_location)).transpose(1,0)
        self.dataidx=dataidx
        self.tissueidx=np.asarray(tissueidx)

    def __getitem__(self, idx):
        # label [B,30+]
        idx=self.dataidx[idx]
        label=torch.load('%s/label_%d.pt'%(data_location,idx))
        label,sitelabel,weight=weightlabel(label)
        sequence=torch.load('%s/sequence_%d.pt'%(data_location,idx))
        if self.radius!=1000:
            sequence=sequence[:,1000-self.radius:1001+self.radius]
        geo=torch.load('%s/geo_%d.pt'%(data_location,idx))
        genelocexp=torch.load('%s/genelocexp_%d.pt'%(data_location,idx))
        return label[:,self.tissueidx],sequence,geo[:,self.tissueidx],self.gene[self.tissueidx],genelocexp[:,self.tissueidx],sitelabel[:,self.tissueidx],weight[:,self.tissueidx]

    def __len__(self):
        return len(self.dataidx)

class testDataset(Dataset):
    def __init__(self, dataprefix='test',radius=1000):
        self.radius=radius
        self.gene=torch.load('%slg2geneexp.pt'%(data_location)).transpose(1,0)
        self.dataprefix=dataprefix
    def __getitem__(self, idx):
        # label [B,1]
        label=torch.load('%s/label_%s_%d.pt'%(data_location,self.dataprefix,idx))
        label,sitelabel,weight=weightlabel(label)
        sequence=torch.load('%s/sequence_%s_%d.pt'%(data_location,self.dataprefix,idx))
        if self.radius!=1000:
            sequence=sequence[:,1000-self.radius:1001+self.radius]
        geo=torch.load('%s/geo_%s_%d.pt'%(data_location,self.dataprefix,idx))
        genelocexp=torch.load('%s/genelocexp_%s_%d.pt'%(data_location,self.dataprefix,idx))
        return label[:,idx:idx+1],sequence,geo,self.gene[idx:idx+1],genelocexp,sitelabel[:,idx:idx+1],weight[:,idx:idx+1]
    def __len__(self):
        return 37
class sitetestDataset(Dataset):
    def __init__(self,radius=1000):
        #use summed label across tissue on purpose
        self.radius=radius
        self.gene=torch.load('%slg2geneexp.pt'%(data_location)).transpose(1,0)
        self.label=torch.load('%s/label_sitetest.pt'%(data_location))
        _,self.sitelabel,self.weight=weightlabel(self.label)
        self.sequence=torch.load('%s/sequence_sitetest.pt'%(data_location))
        if self.radius!=1000:
            self.sequence=self.sequence[:,1000-self.radius:1001+self.radius]
        self.geo=torch.load('%s/geo_sitetest.pt'%(data_location))
        self.genelocexp=torch.load('%s/genelocexp_sitetest.pt'%(data_location))
    def __getitem__(self, idx):
        # label [B,37]
        return self.sitelabel[:,idx:idx+1],self.sequence,self.geo[:,idx:idx+1],self.gene[idx:idx+1],self.genelocexp[:,idx:idx+1],self.sitelabel[:,idx:idx+1],self.weight[:,idx:idx+1]
    def __len__(self):
        return 37
class PLDataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage):
        self.train_dataset = trainDataset(dataidx=np.arange(240),tissueidx=train_list,radius=radius)
        self.valid_dataset = trainDataset(dataidx=np.arange(240,246),radius=radius)
        self.test_dataset = testDataset(dataprefix='test',radius=radius)
        self.sitetest_dataset = sitetestDataset(radius=radius)
        self.tissuetest_dataset = testDataset(dataprefix='tissuetest',radius=radius)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size,shuffle=True)
    def val_dataloader(self):
        return [DataLoader(self.valid_dataset, self.batch_size),DataLoader(self.test_dataset, 1),DataLoader(self.tissuetest_dataset, 1),DataLoader(self.sitetest_dataset, 1)]
    def test_dataloader(self):
        return [DataLoader(self.test_dataset, 1),DataLoader(self.tissuetest_dataset, 1),DataLoader(self.sitetest_dataset, 1)]

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
        self.acc = C.BinaryAccuracy()
        self.ap = C.BinaryAveragePrecision()
        self.mcc = C.BinaryMatthewsCorrCoef()
        self.auc = C.BinaryAUROC()
        self.spec = C.BinarySpecificity()
        self.sens = C.BinaryPrecision()
        self.f1 = C.BinaryF1Score()
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
        # seq [N,501,4]
        # geo [N,24]
        # gene [37,28k]
        # lcoexp [N,2]
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
    def loss_function(self, pred, label,sitelabel,weight) -> dict:
        # label input is 2d [Batch,tissue](0,1) but flattened to calculate 1d binary loss
        # pred also 2d[Batch,3(site,unweighttissue,balancedtissue)],one column is extracted in each subloss calculation

        # weight = weights.view([-1])
        # sitelabel = sitelabel.reshape([-1])
        # pred=pred.view([-1,3])
        generalloss = torch.nn.functional.binary_cross_entropy_with_logits(pred[:,0], label,weight=weight)
        siteloss = torch.nn.functional.binary_cross_entropy_with_logits(pred[:,-1], sitelabel)
        tissueloss = torch.nn.functional.binary_cross_entropy_with_logits(pred[:,1], label,weight=sitelabel)
        balancedtissueloss = torch.nn.functional.binary_cross_entropy_with_logits(pred[:,2], label, weight=weight*sitelabel)
        loss=4*generalloss+2*siteloss+tissueloss+balancedtissueloss
        return {'loss':loss,
                'general_loss':generalloss,
                'site_loss':siteloss,
                'unweighted_tissue_loss':tissueloss,
                'weighted_tissue_loss':balancedtissueloss}
    def shared_step(self, batch):
        y, x, geo, gene, genelocexp,sitelabel,weight = batch
        # batchsize=1 because RAM limit
        y = y.flatten(0,1)  # [50,37]
        x = x.flatten(0,1)  # [50,2001,4]
        geo = geo.flatten(0,1)  # [50,24]
        # gene = gene.flatten(0,1)  # [1,28k,37]
        genelocexp = genelocexp.flatten(0,1).unsqueeze(2)  # [50,2]
        pred = self.forward(x, geo, gene, genelocexp)  # [B,37,3]
        pred=pred.view([-1,4])
        sitelabel=sitelabel.view([-1])
        y=y.view([-1])
        weight=weight.view([-1])
        return pred,y,sitelabel,weight
    def training_step(self, batch, batch_idx):
        self.training = True
        self.train()
        pred,y,sitelabel,weight=self.shared_step(batch)
        loss = self.loss_function(pred, y,sitelabel,weight)
        self.log('train_acc', self.acc(pred[:,0], y), on_epoch=True,on_step=False)
        return loss['loss']
    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.training=False
        self.eval()
        testlabel=['valid','test','tissuetest','sitetest']
        # training_step defines the train loop.
        # it is independent of forward
        if dataloader_idx==0:
            pred,y,sitelabel,weight=self.shared_step(batch)
            loss = self.loss_function(pred, y,sitelabel,weight)
            sitepred = pred[:, -1]
            pred1 = torch.minimum(pred[:, 1],sitepred)
            pred2 = torch.minimum(pred[:, 2],sitepred)
            self.log('valid_loss', loss['loss'], on_epoch=True)
            self.log('valid_site_loss', loss['site_loss'], on_epoch=True)
            self.log('valid_general_loss', loss['general_loss'], on_epoch=True)
            self.log('valid_unweighted_tissue_loss', loss['unweighted_tissue_loss'], on_epoch=True)
            self.log('valid_weighted_tissue_loss', loss['weighted_tissue_loss'], on_epoch=True)
            self.log('valid_acc', self.acc(pred[:, 0], y), on_epoch=True)
            self.log('valid_acc1', self.acc(pred1, y), on_epoch=True)
            self.log('valid_acc2', self.acc(pred2, y), on_epoch=True)
            self.log('valid_accsite', self.acc(sitepred, y), on_epoch=True)
            self.log('valid_accsite2site', self.acc(sitepred, sitelabel), on_epoch=True)
        else:
            pred,y,sitelabel,weight=self.shared_step(batch)
            sitepred = pred[:, -1]
            pred1 = torch.minimum(pred[:, 1],sitepred)
            pred2 = torch.minimum(pred[:, 2],sitepred)
            y = y.view([-1])
            self.log(f"{testlabel[dataloader_idx]}_acc", self.acc(pred[:, 0], y),on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_auc", self.auc(pred[:, 0], y),on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_mcc", self.mcc(pred[:, 0], y),on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_acc_unweighted", self.acc(pred1, y),on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_auc_unweighted", self.auc(pred1, y),on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_mcc_unweighted", self.mcc(pred1, y),on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_acc_weighted", self.acc(pred2, y),on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_auc_weighted", self.auc(pred2, y),on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_mcc_weighted", self.mcc(pred2, y),on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_acc_site", self.acc(sitepred, sitelabel),on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_auc_site", self.auc(sitepred, sitelabel),on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_mcc_site", self.mcc(sitepred, sitelabel),on_epoch=True)
    def test_step(self, batch, batch_idx, dataloader_idx):
        self.training=False
        self.eval()
        testlabel=['test','tissuetest','sitetest']
        # training_step defines the train loop.
        # it is independent of forward
        pred,y,sitelabel,weight=self.shared_step(batch)
        sitepred = pred[:, -1]
        pred1 = torch.minimum(pred[:, 1],sitepred)
        pred2 = torch.minimum(pred[:, 2],sitepred)
        self.log(f"{testlabel[dataloader_idx]}_acc", self.acc(pred[:, 0], y),on_epoch=True,on_step=True)
        self.log(f"{testlabel[dataloader_idx]}_auc", self.auc(pred[:, 0], y),on_epoch=True,on_step=True)
        self.log(f"{testlabel[dataloader_idx]}_mcc", self.mcc(pred[:, 0], y),on_epoch=True,on_step=True)
        self.log(f"{testlabel[dataloader_idx]}_acc_unweighted", self.acc(pred1, y),on_epoch=True,on_step=True)
        self.log(f"{testlabel[dataloader_idx]}_auc_unweighted", self.auc(pred1, y),on_epoch=True,on_step=True)
        self.log(f"{testlabel[dataloader_idx]}_mcc_unweighted", self.mcc(pred1, y),on_epoch=True,on_step=True)
        self.log(f"{testlabel[dataloader_idx]}_acc_weighted", self.acc(pred2, y),on_epoch=True,on_step=True)
        self.log(f"{testlabel[dataloader_idx]}_auc_weighted", self.auc(pred2, y),on_epoch=True,on_step=True)
        self.log(f"{testlabel[dataloader_idx]}_mcc_weighted", self.mcc(pred2, y),on_epoch=True,on_step=True)
        self.log(f"{testlabel[dataloader_idx]}_acc_site", self.acc(sitepred, sitelabel),on_epoch=True,on_step=True)
        self.log(f"{testlabel[dataloader_idx]}_auc_site", self.auc(sitepred, sitelabel),on_epoch=True,on_step=True)
        self.log(f"{testlabel[dataloader_idx]}_mcc_site", self.mcc(sitepred, sitelabel),on_epoch=True,on_step=True)
if __name__ == '__main__':
    if int(args.epoch)==0:
        minepoch=200
        maxepoch=200
    else:
        minepoch=int(args.epoch)
        maxepoch=int(args.epoch)
    # print('model will save to %s/model/%s.pt'%(folder_prefix,prefix))
    checkpoint_callback = ModelCheckpoint(dirpath='%s/model/%s/'%(folder_prefix,prefix))
    logger = CSVLogger("lightning_logs", name=prefix)
    if args.precision!='bf16-mixed':
        args.precision=int(args.precision)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1,
        # benchmark=True,
        logger=logger,
        enable_progress_bar=False,
        max_epochs=maxepoch,
        min_epochs=minepoch,
        precision=args.precision,
        # overfit_batches=2,
        # this will reduce number of samples used, for debugging
        callbacks=[pl.callbacks.EarlyStopping('valid_loss/dataloader_idx_0',patience=50),checkpoint_callback],
        check_val_every_n_epoch=1,
        # limit_val_batches=1,
        # use limit_XXX_batches=? to run part of the sample(for testing later steps)
        # for example when checking gpu usage using log_gpu_memory='all'

        #     default_root_dir=log_path,
        #     auto_scale_batch_size=True,
        #     auto_lr_find='lr',
        #     track_grad_norm='inf',
        #     gradient_clip_val=1, gradient_clip_algorithm="value",
        #     weights_summary='top',
        #     #profiler=pl.profiler.Advanced_profiler
        #     gpus=1,auto_select_gpus=True,
        # use benchmark=True when input size does not change
        # setting deterministic=True give reproducible result but harms performance

    )
    if args.model_path is not None:
        if args.model_path.endswith('pt'):
            model = torch.load(args.model_path)
        elif args.model_path.endswith('ckpt'):
            model = ExpressRM().load_from_checkpoint(args.model_path)
        elif args.model_path.endswith('/'):
            model = ExpressRM().load_from_checkpoint('%s%s' % (args.model_path, os.listdir(args.model_path)[0]))
        else:
            raise NotImplementedError
        model.useseq=useseq
        model.usegeo=usegeo
        model.usegene=usegene
        model.usegenelocexp=usegenelocexp
        model.usetgeo=usetgeo
    else:
        model = ExpressRM(useseq = useseq,usegeo = usegeo,usegene = usegene,usegenelocexp = usegenelocexp,usetgeo = usetgeo)
    data_module = PLDataModule(batch_size=2)
    trainer.fit(model, data_module)
    trainer.test(ckpt_path="best", dataloaders=data_module)
    print(prefix)
    # model.trainer.current_epoch = 0
    # trainer.fit(model, data_module)
    # trainer.test(ckpt_path="best",dataloaders=data_module)
    # trainer.test(ckpt_path="last",dataloaders=data_module)
    # trainer.test(model, data_module)
    # torch.save(model, '%s/model/%s.pt'%(folder_prefix,prefix))
    # print('model saved to %s/model/%s.pt'%(folder_prefix,prefix))