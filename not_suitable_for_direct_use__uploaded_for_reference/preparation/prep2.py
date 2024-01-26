#was in ipynb, copied to py for non-jupyter viewing.
import os
import scipy
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch import nn
dim = 64
import pickle
adaptoutsize=8
import argparse
import sklearn
# print('')
label=pd.read_csv('/data1/yiyou/ExpressRM/selectedlabel.csv',header=0)
tissuelist=label.columns[:-1]
trainidx=np.arange(252009)
np.random.shuffle(trainidx)
testidx=trainidx[:6009]
trainidx=trainidx[6009:]
np.save('/data1/yiyou/ExpressRM/testidx.npy',testidx)
np.save('/data1/yiyou/ExpressRM/trainidx.npy',trainidx)
# trainidx=np.load('/data1/yiyou/ExpressRM/trainidx.npy')
# testidx=np.load('/data1/yiyou/ExpressRM/testidx.npy')

# torch.save(torch.as_tensor(np.sum(1-label,axis=0)/np.sum(label,axis=0)).cuda().float(),'/data1/yiyou/ExpressRM/lossweight.pt')

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
sitelabel=label.iloc[testidx,-1]
assert sum(sitelabel==0)>1000
siteposlabel=label.iloc[testidx[sitelabel>0],:-1]
assert all((siteposlabel>0).sum()>1000)
assert all((siteposlabel==0).sum()>1000)
print(sum(sitelabel==0))
print((siteposlabel>0).sum())
print((siteposlabel==0).sum())
seglength=1000
num_folder=246000//seglength
tissue_specificity_idx_list=[]
test_idx_list=[]
any_pos_idx=testidx[np.sum(label.iloc[testidx]>0,axis=1)>0]
negidx=testidx[np.array(np.sum(label.iloc[testidx],axis=1)==0)]
site_idx=np.append(np.random.permutation(any_pos_idx)[:1000],np.random.permutation(negidx)[:1000])

assert(all(np.sum(label.iloc[negidx],axis=1)==0))

assert(all(np.sum(label.iloc[any_pos_idx],axis=1)>0))
for j in range(37):
    selectedidx=label.loc[any_pos_idx,tissuelist[j]]
    selectedidxpos=any_pos_idx[np.where(selectedidx>0)[0]]
    selectedidxneg=any_pos_idx[np.where(selectedidx==0)[0]]
    tissue_specificity_idx=np.append(np.random.permutation(selectedidxpos)[:1000],np.random.permutation(selectedidxneg)[:1000])
    assert(all(np.sum(label.iloc[tissue_specificity_idx],axis=1)>0))
    tissue_specificity_idx_list.append(tissue_specificity_idx)
    selectedidx=label.loc[testidx,tissuelist[j]]
    selectedidxpos=testidx[np.where(selectedidx>0)[0]]
    selectedidxneg=testidx[np.where(selectedidx==0)[0]]
    test_idx=np.append(np.random.permutation(selectedidxpos)[:1000],np.random.permutation(selectedidxneg)[:1000])
    test_idx_list.append(test_idx)
# np.save('/data1/yiyou/ExpressRM/tissue_specificityidx.npy',any_pos_idx)
# rigidtestidx=np.load('/data1/yiyou/ExpressRM/tissue_specificityidx.npy')
# testidx=np.load('/data1/yiyou/ExpressRM/testidx.npy')
np.save('/data1/yiyou/ExpressRM/tissue_specificityidx.npy',np.array(tissue_specificity_idx_list))
np.save('/data1/yiyou/ExpressRM/testidx_balanced.npy',np.array(test_idx_list))
np.save('/data1/yiyou/ExpressRM/sitetestidx.npy',site_idx)
geneexp=pd.read_csv('/data1/yiyou/ExpressRM/gene_expression/lg2geneexp.csv',index_col='GeneName')
# geneidx=geneexp.index
torch.save(torch.as_tensor(np.asarray(geneexp)).float().cuda(),'/data1/yiyou/ExpressRM/tensor/lg2geneexp.pt')
for i in range(num_folder):
    labelseg=label.iloc[trainidx[seglength*i:seglength*(i+1)]]
    labelseg=torch.as_tensor(np.sign(np.asarray(labelseg))).cuda().float()
    torch.save(labelseg,'/data1/yiyou/ExpressRM/tensor/label_%d.pt'%(i))
labelseg=label.iloc[site_idx]
labelseg=torch.as_tensor(np.sign(np.asarray(labelseg))).cuda().float()
torch.save(labelseg,'/data1/yiyou/ExpressRM/tensor/label_sitetest.pt')
for j in range(37):
    labelseg=label.iloc[tissue_specificity_idx_list[j]]
    labelseg=torch.as_tensor(np.sign(np.asarray(labelseg))).cuda().float()
    torch.save(labelseg,'/data1/yiyou/ExpressRM/tensor/label_tissuetest_%d.pt'%(j))
    labelseg=label.iloc[test_idx_list[j]]
    labelseg=torch.as_tensor(np.sign(np.asarray(labelseg))).cuda().float()
    torch.save(labelseg,'/data1/yiyou/ExpressRM/tensor/label_test_%d.pt'%(j))

seq_list=[]
for seq_record in SeqIO.parse('/data1/yiyou/ExpressRM/selected.fasta',format='fasta'):
    sequence=seq_record.seq
    seq_list.append(sequence)
seq_list=np.asarray(seq_list)
sequence=fasta2binonehot(seq_list)
# np.save('/data1/yiyou/ExpressRM/sequence.npy',sequence)
# sequence=np.load('/data1/yiyou/ExpressRM/sequence.npy')
print(sequence.shape)

for i in range(num_folder):
    sequenceseg=sequence[trainidx[seglength*i:seglength*(i+1)]]
    sequenceseg=torch.as_tensor(sequenceseg).cuda().float()
    torch.save(sequenceseg,'/data1/yiyou/ExpressRM/tensor/sequence_%d.pt'%(i))
sequenceseg=sequence[site_idx]
sequenceseg=torch.as_tensor(sequenceseg).cuda().float()
torch.save(sequenceseg,'/data1/yiyou/ExpressRM/tensor/sequence_sitetest.pt')
for j in range(37):
    sequenceseg=sequence[tissue_specificity_idx_list[j]]
    sequenceseg=torch.as_tensor(sequenceseg).cuda().float()
    torch.save(sequenceseg,'/data1/yiyou/ExpressRM/tensor/sequence_tissuetest_%d.pt'%(j))
    sequenceseg=sequence[test_idx_list[j]]
    sequenceseg=torch.as_tensor(sequenceseg).cuda().float()
    torch.save(sequenceseg,'/data1/yiyou/ExpressRM/tensor/sequence_test_%d.pt'%(j))
###################### Geo #############################
geo_list=[]
geo1=np.asarray(pd.read_csv('/data1/yiyou/ExpressRM/geo/geo.csv',header=0))
for tissue in tissuelist:
    # geo1=np.asarray(pd.read_csv('/data1/yiyou/ExpressRM/geo/geo.csv',header=0))
    geo2=np.asarray(pd.read_csv('/data1/yiyou/ExpressRM/geo/%s.csv'%(tissue),header=0))
    geo=np.append(geo1[:,6:],geo2[:,6:],axis=-1)
    geo_list.append(geo)
geo=np.asarray(geo_list).transpose([1,0,2])
print(geo.shape)
for i in range(num_folder):
    geoseg=geo[trainidx[seglength*i:seglength*(i+1)]]
    geoseg=torch.as_tensor(geoseg).cuda().float()
    torch.save(geoseg,'/data1/yiyou/ExpressRM/tensor/geo_%d.pt'%(i))
geoseg=geo[site_idx]
geoseg=torch.as_tensor(geoseg).cuda().float()
torch.save(geoseg,'/data1/yiyou/ExpressRM/tensor/geo_sitetest.pt')
for j in range(37):
    geoseg=geo[tissue_specificity_idx_list[j],j:j+1]
    geoseg=torch.as_tensor(geoseg).cuda().float()
    torch.save(geoseg,'/data1/yiyou/ExpressRM/tensor/geo_tissuetest_%d.pt'%(j))
    geoseg=geo[test_idx_list[j],j:j+1]
    geoseg=torch.as_tensor(geoseg).cuda().float()
    torch.save(geoseg,'/data1/yiyou/ExpressRM/tensor/geo_test_%d.pt'%(j))
    # print(geo[tissue_specificity_idx_list[j],j:j+1].shape)
genelocexp=np.asarray(pd.read_csv('/data1/yiyou/ExpressRM/gene_expression/lg2hosting_expression.csv'))
for i in range(num_folder):
    tmp=torch.as_tensor(genelocexp[trainidx[i*seglength:(i+1)*seglength]]).float().cuda()
    torch.save(tmp,'/data1/yiyou/ExpressRM/tensor/genelocexp_%d.pt'%(i))
tmp=torch.as_tensor(genelocexp[site_idx]).float().cuda()
print(tmp.shape)
torch.save(tmp,'/data1/yiyou/ExpressRM/tensor/genelocexp_sitetest.pt')

for j in range(37):
    tmp=torch.as_tensor(genelocexp[tissue_specificity_idx_list[j],j:j+1]).float().cuda()
    torch.save(tmp,'/data1/yiyou/ExpressRM/tensor/genelocexp_tissuetest_%d.pt'%(j))
    tmp=torch.as_tensor(genelocexp[test_idx_list[j],j:j+1]).float().cuda()
    torch.save(tmp,'/data1/yiyou/ExpressRM/tensor/genelocexp_test_%d.pt'%(j))