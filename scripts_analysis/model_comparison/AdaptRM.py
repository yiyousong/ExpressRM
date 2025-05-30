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
from torch.utils.tensorboard import SummaryWriter

# define directory
folder_prefix = './'
data_location = './data'

# configuration parameters
dim = 64
droprate = 0.25
adaptoutsize = 15
geneinputsize = 28278
genelocinputsize = geneinputsize

# initialize 
writer = SummaryWriter()
torch.set_num_threads(10)

# argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None, help='add path to continue training else start anew')
parser.add_argument('--seq', default=True, help='1 for true,default false')
parser.add_argument('--gene', default=False, help='')
parser.add_argument('--genelocexp', default=False, help='')
parser.add_argument('--geo', default=False, help='')
parser.add_argument('--tgeo', default=False, help='')
parser.add_argument('--featurelist', default=None, help='1,0,0,0,0\n sequence,gene,genelocexp,geo,tgeo')
parser.add_argument('--radius', default=1000, help='2*radius+1')
parser.add_argument('--epoch', default=0, help='')
parser.add_argument('--prefix', default='AdaptRM', help='')
parser.add_argument('--autoprefix', default=True, help='')
parser.add_argument('--trainlist', default=None, help='use testlist')
parser.add_argument('--testlist', default=None, help='drop idx (during training) or leaveoneout')
parser.add_argument('--precision', default=32, help='drop idx (during training) or leaveoneout')
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
    prefix += '__AdaptRM'
in_chan = 4
prefix += '_%dbp' % (2 * radius + 1)

def weightlabel(labelwithsite):
    label=labelwithsite[...,:-1]
    sitelabel = labelwithsite[...,-1:]
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
        self.gene=torch.load('%s/gene_expression/lg2geneexp.pt'%(data_location)).transpose(1,0)
        self.dataidx=dataidx
        self.tissueidx=np.asarray(tissueidx)

    def __getitem__(self, idx):
        # label [B,30+]
        idx=self.dataidx[idx]
        label=torch.load('%s/train/label_%d.pt'%(folder_prefix,idx))
        label,sitelabel,weight=weightlabel(label)
        sequence=torch.load('%s/sequence/sequence_%d.pt'%(data_location,idx))
        if self.radius!=1000:
            sequence=sequence[:,1000-self.radius:1001+self.radius]
        geo=torch.load('%s/geo/geo_%d.pt'%(data_location,idx))
        genelocexp=torch.load('%s/gene_expression/genelocexp_%d.pt'%(data_location,idx))
        return label[:,self.tissueidx],sequence,geo[:,self.tissueidx],self.gene[self.tissueidx],genelocexp[:,self.tissueidx],sitelabel,weight[:,self.tissueidx],idx

    def __len__(self):
        return len(self.dataidx)

class testDataset(Dataset):
    def __init__(self, dataprefix='test',radius=1000):
        self.radius=radius
        self.gene=torch.load('%s/gene_expression/lg2geneexp.pt'%(data_location)).transpose(1,0)
        self.dataprefix=dataprefix
    def __getitem__(self, idx):
        label=torch.load('%s/train/label_%s_%d.pt'%(folder_prefix,self.dataprefix,idx))
        label,sitelabel,weight=weightlabel(label)
        sequence=torch.load('%s/sequence/sequence_%s_%d.pt'%(data_location,self.dataprefix,idx))
        if self.radius!=1000:
            sequence=sequence[:,1000-self.radius:1001+self.radius]
        geo=torch.load('%s/geo/geo_%s_%d.pt'%(data_location,self.dataprefix,idx))
        genelocexp=torch.load('%s/gene_expression/genelocexp_%s_%d.pt'%(data_location,self.dataprefix,idx))
        return label[:,idx:idx+1],sequence,geo,self.gene[idx:idx+1],genelocexp,sitelabel,weight[:,idx:idx+1],idx
    def __len__(self):
        return 37

class sitetestDataset(Dataset):
    def __init__(self,radius=1000):
        #use summed label across tissue on purpose
        self.radius=radius
        self.gene=torch.load('%s/gene_expression/lg2geneexp.pt'%(data_location)).transpose(1,0)
        self.label=torch.load('%s/train/label_sitetest.pt'%(folder_prefix))
        _,self.sitelabel,self.weight=weightlabel(self.label)
        self.sequence=torch.load('%s/sequence/sequence_sitetest.pt'%(data_location))
        if self.radius!=1000:
            self.sequence=self.sequence[:,1000-self.radius:1001+self.radius]
        self.geo=torch.load('%s/geo/geo_sitetest.pt'%(data_location))
        self.genelocexp=torch.load('%s/gene_expression/genelocexp_sitetest.pt'%(data_location))
    def __getitem__(self, idx):
        # label [B,37]
        return self.sitelabel,self.sequence,self.geo[:,idx:idx+1],self.gene[idx:idx+1],self.genelocexp[:,idx:idx+1],self.sitelabel,self.weight[:,idx:idx+1],idx
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

class AdaptRM(pl.LightningModule):
    def __init__(self, patchsize=7, patchstride=5, inchan=4, dim=64, kernelsize=7,
                 adaptoutsize=9, geneoutsize=32, geooutsize=32, droprate=0.25, lr=2e-5):
        super(AdaptRM, self).__init__()
        self.useseq = useseq
        self.acc = C.BinaryAccuracy()
        self.ap = C.BinaryAveragePrecision()
        self.mcc = C.BinaryMatthewsCorrCoef()
        self.auc = C.BinaryAUROC()
        self.spec = C.BinarySpecificity()
        self.sens = C.BinaryPrecision()
        self.f1 = C.BinaryF1Score()
        self.learning_rate = lr
        self.save_hyperparameters()
        self.model = nn.Sequential(
                            nn.Conv1d(4, 64, 11,5), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.Dropout(droprate),
                              nn.Conv1d(64,128,7),nn.BatchNorm1d(128), nn.LeakyReLU(),nn.Dropout(droprate),
                              nn.Conv1d(128,128,7),nn.BatchNorm1d(128), nn.LeakyReLU(),nn.Dropout(droprate),
                              nn.AdaptiveAvgPool1d(19),
                              nn.Conv1d(128,64,7),nn.BatchNorm1d(64), nn.LeakyReLU(),nn.Dropout(droprate),
                              nn.Conv1d(64,64,7),nn.BatchNorm1d(64), nn.LeakyReLU(),nn.Dropout(droprate),
            nn.Flatten(-2),
                              nn.Linear(7 * 64,1000),nn.LeakyReLU(),nn.Dropout(droprate),nn.Linear(1000,38)
                              )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x, **kwargs):
        out = self.model(x)
        return out

    def loss_function(self, pred, label) -> dict:
        weight=(label*2+1)
        weight[:,-1]=2.0
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label,weight=weight)
        return loss

    def shared_step(self, batch):
        y, x, geo, gene, genelocexp, sitelabel, weight, idx = batch
        # batchsize=1 because RAM limit
        y = y.flatten(0, 1)  # [50,37]
        x = x.flatten(0, 1).transpose(-1, -2)  # [50,2001,4]
        pred = self.forward(x)  # [B,37,3]
        return pred, y, idx, sitelabel

    def training_step(self, batch, batch_idx):
        self.training = True
        self.train()
        pred, y, idx, sitelabel = self.shared_step(batch)
        sitelabel=sitelabel.flatten(0,1)
        y=torch.cat([y,sitelabel],dim=-1)
        loss = self.loss_function(pred, y)
        self.log('train_acc', self.acc(pred, y), on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.training = False
        self.eval()
        testlabel = ['valid', 'test', 'tissuetest', 'sitetest']
        # training_step defines the train loop.
        # it is independent of forward
        pred, y, idx,sitelabel = self.shared_step(batch)
        if dataloader_idx == 0:
            pred=pred[:,:-1]
            loss = self.loss_function(pred, y)
            self.log('valid_loss', loss, on_epoch=True)
            self.log('valid_acc', self.acc(pred, y), on_epoch=True)
        elif dataloader_idx == 3:
            pred=pred[:,-1]
            sitelabel=sitelabel[0,:,0]
            self.log(f"{testlabel[dataloader_idx]}_acc", self.acc(pred, sitelabel), on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_auc", self.auc(pred, sitelabel), on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_mcc", self.mcc(pred, sitelabel), on_epoch=True)
        else:
            pred = pred[:, idx:idx + 1]
            self.log(f"{testlabel[dataloader_idx]}_acc", self.acc(pred, y), on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_auc", self.auc(pred, y), on_epoch=True)
            self.log(f"{testlabel[dataloader_idx]}_mcc", self.mcc(pred, y), on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx):
        self.training = False
        self.eval()
        testlabel = ['test', 'tissuetest', 'sitetest']
        # training_step defines the train loop.
        # it is independent of forward
        pred, y, idx,sitelabel = self.shared_step(batch)

        if dataloader_idx == 2:
            pred=pred[:,-1]
            sitelabel=sitelabel[0,:,0]
            self.log(f"{testlabel[dataloader_idx]}_acc", self.acc(pred, sitelabel), on_epoch=True, on_step=True)
            self.log(f"{testlabel[dataloader_idx]}_auc", self.auc(pred, sitelabel), on_epoch=True, on_step=True)
            self.log(f"{testlabel[dataloader_idx]}_mcc", self.mcc(pred, sitelabel), on_epoch=True, on_step=True)
        else:
            pred = pred[:, idx:idx + 1]
            self.log(f"{testlabel[dataloader_idx]}_acc", self.acc(pred, y), on_epoch=True, on_step=True)
            self.log(f"{testlabel[dataloader_idx]}_auc", self.auc(pred, y), on_epoch=True, on_step=True)
            self.log(f"{testlabel[dataloader_idx]}_mcc", self.mcc(pred, y), on_epoch=True, on_step=True)

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
        callbacks=[pl.callbacks.EarlyStopping('valid_loss/dataloader_idx_0',patience=100),checkpoint_callback],
        check_val_every_n_epoch=1,
        # limit_val_batches=1,
        # use limit_XXX_batches=? to run part of the sample(for testing later steps)
        # for example when checking gpu usage using log_gpu_memory='all'
        # default_root_dir=log_path,
        # auto_scale_batch_size=True,
        # auto_lr_find='lr',
        # track_grad_norm='inf',
        # gradient_clip_val=1, gradient_clip_algorithm="value",
        # weights_summary='top',
        # profiler=pl.profiler.Advanced_profiler
        # gpus=1,auto_select_gpus=True,
        # use benchmark=True when input size does not change
        # setting deterministic=True give reproducible result but harms performance
    )

    model = AdaptRM()
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