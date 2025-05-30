import numpy as np
import copy
from torch import nn
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

def evaluate(pred, y, prefix='', trim=0):
    # assume label to be [N]
    pred = np.squeeze(pred.cpu().detach().numpy())
    y = np.squeeze(y.cpu().detach().numpy())
    predint = (pred > 0).astype(int)
    if trim != 0:
        posidx = np.where(y > 0)[:trim]
        negidx = np.where(y == 0)[:trim]
        idx = np.append(posidx, negidx)
        y = y[idx]
        pred = pred[idx]
    y = y.astype(int)
    p = y[pred > 0]
    n = y[pred < 0]
    tp = np.sum(p)
    fn = np.sum(1 - p)
    tn = np.sum(1 - n)
    fp = np.sum(n)
    sense = tp / (fn + tp + 1e-4)
    speci = tn / (tn + fp + 1e-4)
    auc = metrics.roc_auc_score(y, pred)
    if np.sum(predint) * np.sum(1 - predint) != 0:
        f1 = metrics.f1_score(y, predint)
        mcc = metrics.matthews_corrcoef(y, predint)
    else:
        f1 = mcc = 0
    pr = metrics.average_precision_score(y, pred)
    result = np.array([auc, f1, mcc, pr, sense, speci])
    # result.append(auc)
    print('%s:\tauc: %.3f\tf1: %.3f\tmcc: %.3f\tap: %.3f\tsenseticity: %.3f\tspecificity: %.3f\t' % (
    prefix, auc, f1, mcc, pr, sense, speci), flush=True)
    return result

class Residual(nn.Module):
    ##### Residual on raw weighted sum, performs BatchNorm,Activation,model #####
    def __init__(self,model,append_dim=1,bypass=nn.Identity()):
        super(Residual, self).__init__()
        self.model=model
        self.bypass=bypass
        self.append_dim=append_dim
    def forward(self,x):
        out=self.model(x)
        bypass=self.bypass(x)
        if out.size()==bypass.size():
            out+=bypass
        else:
            out=torch.cat([out,bypass],dim=self.append_dim)
        return out
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

        self.fakemodel=nn.Identity()
    def forward(self, x):
        return x.view(*self.shape)

class Show(nn.Module):
    def __init__(self):
        super(Show, self).__init__()

    def forward(self, x):
        print(x.size())
        return x


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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
class MultiModelAdaptPooling(nn.Module):
    def __init__(self, model, outsizelist=np.array([9, 25, 64])):
        super(MultiModelAdaptPooling, self).__init__()
        self.modellist = []
        for i in outsizelist:
            self.modellist.append(
                nn.Sequential(nn.AdaptiveAvgPool1d(i),
                              copy.deepcopy(model)).cuda())

    def forward(self, x):
        outlist = []
        for model in self.modellist:
            outlist.append(model(x))
        out = torch.cat(outlist, -1)
        return out
