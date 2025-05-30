import pandas as pd
import numpy as np
import torch
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
# import transformer as t
import itertools
import argparse


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features).cuda(),requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(features).cuda(),requires_grad=True)
        self.eps = eps

    def forward(self, x):
        # print(x.size())
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ConvMixer(nn.Module):
    def __init__(self, vocab=4, dim=64, dropout=0.1,sephead=False, adaptout=7, depth=5, kernel_size=7, patch_size=7, n_classes=100,ac1='relu',ac2='relu'):
        super(ConvMixer, self).__init__()
        self.model = nn.Sequential(
            Conv1dtranspose(vocab, dim, dropout=dropout, in_transpose=True, kernel_size=patch_size,
                            stride=patch_size - 2,acti=ac1),
            *[nn.Sequential(
                Residual(Conv1dtranspose(dim, dim,acti=ac1, dropout=dropout, kernel_size=kernel_size, groups=dim,
                                         padding=int(kernel_size // 2))),
                Conv1dtranspose(dim, dim,acti=ac1, kernel_size=1, dropout=dropout),
                # Residual(SelfAttention(dim)),

            ) for i in range(depth)],
            nn.AdaptiveAvgPool1d(adaptout),
            Linout(in_size=dim * adaptout, out_size=n_classes,sephead=sephead,acti=ac2)
        )

    def forward(self, x):
        return self.model(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # self.norm = LayerNorm(layer.size)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return x


# class Decoder(nn.Module):
#     "Core encoder is a stack of N layers"
#
#     def __init__(self, layer, N):
#         super(Decoder, self).__init__()
#         self.layers = clones(layer, N)
#         # self.norm = LayerNorm(layer.size)
#
#     def forward(self, m):
#         "Pass the input (and mask) through each layer in turn."
#         x = m
#         for layer in self.layers:
#             x = layer(x, m)
#         return x


class SelfAttention(nn.Module):
    def __init__(self, inputsize, headnum=8, modelsize=None):
        super(SelfAttention, self).__init__()
        if modelsize is None:
            modelsize = inputsize // headnum
        self.Wq = clones(nn.Linear(inputsize, modelsize, bias=False), headnum)
        self.Wk = clones(nn.Linear(inputsize, modelsize), headnum)
        self.Wv = clones(nn.Linear(inputsize, modelsize), headnum)
        self.size = 1 / (modelsize ** 0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, m=None):
        z = []
        if m is None:
            m = x
        for i in range(len(self.Wq)):
            q = self.Wq[i](x)
            k = self.Wk[i](m).transpose(-1, -2)
            weight = torch.mul(torch.matmul(q, k), self.size)
            v = torch.matmul(self.softmax(weight), self.Wv[i](m))
            z.append(v)
        z = torch.cat(z, -1)
        return z


# class rfft2(nn.Module):
#     def __init__(self):
#         super(rfft2, self).__init__()
#
#     def forward(self, x):
#         return torch.fft.rfft2(x)


class Encoderlayer(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, inputsize, outsize, dropout=0.2, modelsize=None,acti='relu', headnum=16, fourier=False, res=False):
        super(Encoderlayer, self).__init__()
        if modelsize is None:
            self.res = True
            modelsize = int(inputsize / headnum)
        else:
            self.res = False
        if modelsize * headnum == outsize:
            self.resout = True
        else:
            self.resout = False
        if fourier:
            self.att = rfft2()
        else:
            self.att = SelfAttention(inputsize, headnum, modelsize)
        self.Wz = nn.Sequential(nn.Linear(modelsize * headnum, outsize), actfunc(acti), nn.Linear(outsize, outsize))
        self.sublayer = [LayerNorm(inputsize), LayerNorm(modelsize * headnum)]
        self.dropout = clones(nn.Dropout(dropout), 3)

    def forward(self, x):
        if self.res:
            z = x + self.dropout[0](self.att(self.sublayer[0](x)))
        else:
            z = self.att(x)
        if self.resout:
            out = z + self.dropout[1](self.Wz(self.sublayer[1](z)))
        else:
            out = self.Wz(z)
        return out
class ReshapeTrim(nn.Module):
    def __init__(self,width=40,dim=-2):
        super(ReshapeTrim, self).__init__()
        self.width=width
        self.dim=dim
    def forward(self,x,width=None,dim=None):
        if width is None:
            width=self.width
        if dim is None:
            dim=self.dim
        numpiece=x.size()[dim]//width
        start=x.size()[dim]%width//2
        x=x.narrow(dim,start,numpiece*width)
        x=x.view([numpiece, width, -1])
        return x
#
# class Decoderlayer(nn.Module):
#     "Core encoder is a stack of N layers"
#
#     def __init__(self, inputsize, outsize, dropout=0.2, modelsize=None, headnum=16, fourier=False, res=False):
#         super(Decoderlayer, self).__init__()
#         if modelsize is None:
#             self.res = True
#             modelsize = int(inputsize / headnum)
#         else:
#             self.res = False
#         if modelsize * headnum == outsize:
#             self.resout = True
#         else:
#             self.resout = False
#         if fourier:
#             self.att = rfft2()
#         else:
#             self.att = SelfAttention(inputsize, headnum, modelsize)
#         self.decatt = SelfAttention(headnum * modelsize, headnum, modelsize)
#         self.Wz = nn.Sequential(nn.Linear(modelsize * headnum, outsize), nn.PReLU(), nn.Linear(outsize, outsize))
#         self.sublayer = [LayerNorm(inputsize), LayerNorm(modelsize * headnum), LayerNorm(modelsize * headnum)]
#         self.dropout = clones(nn.Dropout(dropout), 3)
#
#     def forward(self, x, m):
#         if self.res:
#             z = x + self.dropout[0](self.att(self.sublayer[0](x)))
#         else:
#             z = self.att(x)
#         z = z + self.dropout[1](self.decatt(self.sublayer[1](z), m))
#         if self.resout:
#             out = z + self.dropout[2](self.Wz(self.sublayer[2](z)))
#         else:
#             out = self.Wz(z)
#         return out


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, len=None, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.len = len
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # pe = torch.zeros(max_len, 1)
        # x=torch.cat([x,self.pe[:x.size(-2)]],dim=-2)
        position = torch.arange(0, max_len).unsqueeze(1)
        position *= 2
        div_term = torch.exp(position / d_model * math.log(10000))
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        # pe = pe.unsqueeze(0)
        pe.requires_grad = False
        pe = pe / 10
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(-2)]
        return x


class AttentionMerge(nn.Module):
    def __init__(self, out_size,  in_size=None,d_model=256,acti='relu', dropout=0.2):
        super(AttentionMerge, self).__init__()
        self.size = 1 / (d_model ** 0.5)
        self.att_v = nn.Sequential(nn.Linear(in_size, d_model),actfunc(acti))
        self.att_u = nn.Sequential(nn.Linear(in_size, d_model), nn.Tanh())
        self.att=nn.Sequential(nn.Linear(d_model,1),nn.Softmax(dim=0))
        self.out = nn.Sequential(nn.Linear(in_size, d_model), actfunc(acti), nn.Dropout(dropout),
                                 nn.Linear(d_model, out_size),nn.Sigmoid())

    def forward(self, x):
        x2 = torch.flatten(x, 1)
        v = self.att_v(x2)
        u = self.att_v(x2)
        weight = self.att(v*u)
        out = self.out(torch.matmul(torch.transpose(weight, 0, 1), x2))
        return out


class Residual(nn.Module):
    def __init__(self, fn, dim=64):
        super().__init__()
        self.fn = fn
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.fn(x) + x


class Conv1dtranspose(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, dilation=1, padding=0, pooling=False,
                 in_transpose=False, out_transpose=False, groups=1, dropout=0.1,acti='relu'):
        super(Conv1dtranspose, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels=in_chan, out_channels=out_chan, padding=padding, groups=groups,
                              kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.in_transpose = in_transpose
        self.out_transpose = out_transpose
        # self.bn = nn.LayerNorm(out_chan)
        self.dropout = nn.Dropout(dropout)
        self.out=actfunc(acti)
        self.pooling = pooling
        if pooling:
            self.pool = nn.MaxPool1d(2)

    def forward(self, x, in_transpose=False):
        if in_transpose:
            x = torch.transpose(x, -1, -2)
        elif self.in_transpose:
            x = torch.transpose(x, -1, -2)
        x = self.conv(x)
        # x = self.bn(x)
        x = self.out(self.dropout(x))
        if self.pooling:
            x = self.pool(x)
        if self.out_transpose:
            x = torch.transpose(x, -1, -2)
        return x


def losswithmask(pred, labelsign, loss_function=nn.functional.binary_cross_entropy_with_logits):
    mask = torch.square(labelsign)
    label = (labelsign + 1) / 2
    label = label[mask>0]
    pred=pred[mask>0]
    label.requires_grad = False
    mask.requires_grad = False
    # loss=loss_function(pred,label)
    loss = loss_function(pred, label, weight=mask, reduction='sum')
    # loss=torch.sum(loss*mask)/torch.sum(mask)
    return loss


# def weak_loss(pred,label,loss_func):
#     predsign=torch.squeeze(label)-torch.sign(torch.sum(pred>0))
#     size=pred.size()[0]
#     labelsize=pred.size()[1]
#     ones=torch.as_tensor(np.ones([size,labelsize]))
#     zeros=torch.as_tensor(np.ones([size,labelsize])*-1)
#
#     tmp=ones*(predsign==1)
#     tmp+=zeros*(predsign==-1)
#     loss=losswithmask(pred,tmp,loss_func)
#     return loss
def actfunc(activation='relu'):
    if activation == 'relu':
        act = nn.ReLU
    elif activation == 'elu':
        act = nn.ELU
    else:
        act = nn.PReLU
    return act()
class CLS(nn.Module):
    def __init__(self, in_size, out_size=100,acti='relu'):
        super(CLS, self).__init__()
        self.model = nn.Sequential(nn.Linear(in_size, 1000),
                                   actfunc(acti),
                                   nn.Linear(1000, out_size), nn.Sigmoid())

    def forward(self, x):
        x = x[0]
        out = self.model(x).view([1,-1])
        return out


class Transformer(nn.Module):
    def __init__(self, N=5, src_vocab=64,d_model=256, dropout=0.25, h=8,outsize=100,acti='relu'):
        super(Transformer, self).__init__()
        self.Embed = nn.Sequential(
            Conv1dtranspose(src_vocab, d_model,acti=acti, in_transpose=True, out_transpose=True, kernel_size=7, stride=5),
            nn.Flatten(0, 1))
        self.model = nn.Sequential(
            PositionalEncoding(d_model, dropout),
            Encoder(Encoderlayer(inputsize=d_model, outsize=d_model, headnum=h, dropout=dropout,acti=acti), N),
            CLS(d_model,out_size=outsize,acti=acti)
        )
        self.cls = (torch.ones([1, d_model]) * -1)
        self.cls.requires_grad = False
        self.cls = self.cls.cuda()

    def forward(self, x):
        x = self.Embed(x)
        x = torch.cat([self.cls, x], dim=0)
        return self.model(x)
class Mergemodel(nn.Module):
    def __init__(self,modelhead,modellist):
        super(Mergemodel, self).__init__()
        self.modelhead=modelhead
        self.modellist=modellist
    def forward(self,xlist):
        outlist=[]
        multi_input=False
        if isinstance(xlist,list):
            multi_input=True
        for i in range(len(self.modellist)):
            if multi_input:
                x=xlist[i]
            else:
                x=xlist
            out=self.modellist[i].forward(x)
            outlist.append(out)

        return self.modelhead(torch.cat(outlist,dim=-1))
class Linout(nn.Module):
    def __init__(self,in_size,out_size,hidden=1000,acti='relu',dropout=0.2,sephead=False):
        super(Linout, self).__init__()
        self.flat=nn.Flatten()
        if not sephead:
            self.model = nn.Sequential(
                nn.Linear(in_size, hidden),
                # nn.BatchNorm1d(hidden),
                actfunc(acti),
                nn.Dropout(dropout),
                nn.Linear( hidden, out_size),nn.Sigmoid())
        else:
            hidden=hidden//16
            self.modelpart = nn.Sequential(
                nn.Linear(in_size, hidden),
                # nn.BatchNorm1d(hidden),
                actfunc(acti),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_size), nn.Sigmoid())
            self.model=list2model(clones(self.modelpart,out_size))
    def forward(self,x):
        x=self.flat(x)
        out=self.model(x)
        return out
class list2model(nn.Module):
    def __init__(self, modellist):
        super(list2model, self).__init__()
        self.model=modellist
    def forward(self,x):
        out_list=[]
        for model in self.model:
            out=model(x)
            out_list.append(out)
        out=torch.cat(out_list)
        return out



def np2tensor(input_list, label_list=None, mer=1,newvar=0, chunk=False,instance_length=40,shuffle=False):
    input_tensor_list = []
    if label_list is not None:
        label_tensor_list = []
    index = np.arange(len(input_list))
    if shuffle:
        np.random.shuffle(index)
    for i in index:
        if label_list is not None:
            label = np.reshape(np.array(label_list[i]),[1,-1])
        input = np.squeeze(np.int32(input_list[i]))
        input = np.reshape(input, [-1, 4])
        index1 = np.where(input == 1)[-1]
        if mer > 1:
            idx = index1[mer - 1:]
            for k in range(mer - 1):
                idx = idx + (4 ** (mer - 1 - k)) * index1[k:k - mer + 1]
            data = np.zeros([input.shape[0] - mer + 1, 4 ** mer + newvar])
            data[np.where(input == 1)[0][:-mer + 1], idx] = 1
        else:
            data=input
        if chunk:
            data = data[:(len(data) // (instance_length - mer + 1)) * (instance_length - mer + 1)]
            data = np.reshape(data, [-1, instance_length - mer + 1, 4 ** mer + newvar])
        else:
            if mer >1:
                data = np.reshape(data, [1, -1, 4 ** mer + newvar])
            else:
                data = np.reshape(input, [1, -1, 4])
                if newvar > 0:
                    data = np.append(data, np.zeros([1, data.shape[1], newvar]), axis=-1)
        input = torch.as_tensor(data).float().cuda()
        input_tensor_list.append(input)
        if label_list is not None:
            label = torch.as_tensor(label).float().cuda()
            label_tensor_list.append(label)
    if label_list is not None:
        return input_tensor_list, label_tensor_list
    else:
        return input_tensor_list


