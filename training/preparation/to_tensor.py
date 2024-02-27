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
parser = argparse.ArgumentParser()
parser.add_argument('--mainfolder',default=os.path.dirname(os.path.abspath(__file__)),help='folder to dump everything (with/ at end)')
### must change the following path ###
folder_path=args.mainfolder[:-1]

os.system('mkdir -p %s/tensor'%(folder_path))
label=pd.read_csv('%s/selectedlabel.csv'%(folder_path),header=0)
tissuelist=label.columns[:-1]
trainidx=np.arange(252009)
np.random.shuffle(trainidx)
testidx=trainidx[:6009]
trainidx=trainidx[6009:]
np.save('%s/testidx.npy'%(folder_path),testidx)
np.save('%s/trainidx.npy'%(folder_path),trainidx)
# trainidx=np.load('%s/trainidx.npy'%(folder_path))
# testidx=np.load('%s/testidx.npy'%(folder_path))

# torch.save(torch.as_tensor(np.sum(1-label,axis=0)/np.sum(label,axis=0)).cuda().float(),'%s/lossweight.pt'%(folder_path))

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
    
### generating index. ###

regenerate=True
while regenerate:
    sitelabel=label.iloc[testidx,-1]
    siteposlabel=label.iloc[testidx[sitelabel>0],:-1]
    # print(sum(sitelabel==0))
    # print((siteposlabel>0).sum())
    # print((siteposlabel==0).sum())
    seglength=1000
    num_folder=246000//seglength
    tissue_specificity_idx_list=[]
    test_idx_list=[]
    any_pos_idx=testidx[np.sum(label.iloc[testidx]>0,axis=1)>0]
    negidx=testidx[np.array(np.sum(label.iloc[testidx],axis=1)==0)]
    site_idx=np.append(np.random.permutation(any_pos_idx)[:1000],np.random.permutation(negidx)[:1000])
    try:
        assert sum(sitelabel==0)>1000
        assert all((siteposlabel>0).sum()>1000)
        assert all((siteposlabel==0).sum()>1000)
        assert(all(np.sum(label.iloc[negidx],axis=1)==0))
        assert(all(np.sum(label.iloc[any_pos_idx],axis=1)>0))
        regenerate=False
    except:
        pass    

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
np.save('%s/tissue_specificityidx.npy'%(folder_path),np.array(tissue_specificity_idx_list))
np.save('%s/testidx_balanced.npy'%(folder_path),np.array(test_idx_list))
np.save('%s/sitetestidx.npy'%(folder_path),site_idx)

### saving gene expression file to tensor ### 

geneexp=pd.read_csv('%s/gene_expression/lg2geneexp.csv'%(folder_path),index_col='GeneName')
# geneidx=geneexp.index
torch.save(torch.as_tensor(np.asarray(geneexp)).float().cuda(),'%s/tensor/lg2geneexp.pt'%(folder_path))

### segmenting label and saving to tensor ### 

for i in range(num_folder):
    labelseg=label.iloc[trainidx[seglength*i:seglength*(i+1)]]
    labelseg=torch.as_tensor(np.sign(np.asarray(labelseg))).cuda().float()
    torch.save(labelseg,'%s/tensor/label_%d.pt'%(folder_path,i))
labelseg=label.iloc[site_idx]
labelseg=torch.as_tensor(np.sign(np.asarray(labelseg))).cuda().float()
torch.save(labelseg,'%s/tensor/label_sitetest.pt'%(folder_path))
for j in range(37):
    labelseg=label.iloc[tissue_specificity_idx_list[j]]
    labelseg=torch.as_tensor(np.sign(np.asarray(labelseg))).cuda().float()
    torch.save(labelseg,'%s/tensor/label_tissuetest_%d.pt'%(folder_path,j))
    labelseg=label.iloc[test_idx_list[j]]
    labelseg=torch.as_tensor(np.sign(np.asarray(labelseg))).cuda().float()
    torch.save(labelseg,'%s/tensor/label_test_%d.pt'%(folder_path,j))

### processing fasta sequence and saving to tensor ### 

seq_list=[]
for seq_record in SeqIO.parse('%s/selected.fasta'%(folder_path),format='fasta'):
    sequence=seq_record.seq
    seq_list.append(sequence)
seq_list=np.asarray(seq_list)
sequence=fasta2binonehot(seq_list)
np.save('%s/sequence.npy'%(folder_path),sequence)
# sequence=np.load('%s/sequence.npy'%(folder_path))
# print(sequence.shape)
for i in range(num_folder):
    sequenceseg=sequence[trainidx[seglength*i:seglength*(i+1)]]
    sequenceseg=torch.as_tensor(sequenceseg).cuda().float()
    torch.save(sequenceseg,'%s/tensor/sequence_%d.pt'%(folder_path,i))
sequenceseg=sequence[site_idx]
sequenceseg=torch.as_tensor(sequenceseg).cuda().float()
torch.save(sequenceseg,'%s/tensor/sequence_sitetest.pt'%(folder_path))
for j in range(37):
    sequenceseg=sequence[tissue_specificity_idx_list[j]]
    sequenceseg=torch.as_tensor(sequenceseg).cuda().float()
    torch.save(sequenceseg,'%s/tensor/sequence_tissuetest_%d.pt'%(folder_path,j))
    sequenceseg=sequence[test_idx_list[j]]
    sequenceseg=torch.as_tensor(sequenceseg).cuda().float()
    torch.save(sequenceseg,'%s/tensor/sequence_test_%d.pt'%(folder_path,j))
    
### saving geographic encoding to tensor ###

geo_list=[]
geo1=np.asarray(pd.read_csv('%s/geo/geo.csv'%(folder_path),header=0))
for tissue in tissuelist:
    # geo1=np.asarray(pd.read_csv('%s/geo/geo.csv'%(folder_path),header=0))
    geo2=np.asarray(pd.read_csv('%s/geo/%s.csv'%(folder_path,tissue),header=0))
    geo=np.append(geo1[:,6:],geo2[:,6:],axis=-1)
    geo_list.append(geo)
geo=np.asarray(geo_list).transpose([1,0,2])
for i in range(num_folder):
    geoseg=geo[trainidx[seglength*i:seglength*(i+1)]]
    geoseg=torch.as_tensor(geoseg).cuda().float()
    torch.save(geoseg,'%s/tensor/geo_%d.pt'%(folder_path,i))
geoseg=geo[site_idx]
geoseg=torch.as_tensor(geoseg).cuda().float()
torch.save(geoseg,'%s/tensor/geo_sitetest.pt'%(folder_path))
for j in range(37):
    geoseg=geo[tissue_specificity_idx_list[j],j:j+1]
    geoseg=torch.as_tensor(geoseg).cuda().float()
    torch.save(geoseg,'%s/tensor/geo_tissuetest_%d.pt'%(folder_path,j))
    geoseg=geo[test_idx_list[j],j:j+1]
    geoseg=torch.as_tensor(geoseg).cuda().float()
    torch.save(geoseg,'%s/tensor/geo_test_%d.pt'%(folder_path,j))
    # print(geo[tissue_specificity_idx_list[j],j:j+1].shape)

### saving hosting gene expression to tensor ###

genelocexp=np.asarray(pd.read_csv('%s/gene_expression/lg2hosting_expression.csv'%(folder_path)))
for i in range(num_folder):
    tmp=torch.as_tensor(genelocexp[trainidx[i*seglength:(i+1)*seglength]]).float().cuda()
    torch.save(tmp,'%s/tensor/genelocexp_%d.pt'%(folder_path,i))
tmp=torch.as_tensor(genelocexp[site_idx]).float().cuda()
print(tmp.shape)
torch.save(tmp,'%s/tensor/genelocexp_sitetest.pt'%(folder_path))

for j in range(37):
    tmp=torch.as_tensor(genelocexp[tissue_specificity_idx_list[j],j:j+1]).float().cuda()
    torch.save(tmp,'%s/tensor/genelocexp_tissuetest_%d.pt'%(folder_path,j))
    tmp=torch.as_tensor(genelocexp[test_idx_list[j],j:j+1]).float().cuda()
    torch.save(tmp,'%s/tensor/genelocexp_test_%d.pt'%(folder_path,j))
