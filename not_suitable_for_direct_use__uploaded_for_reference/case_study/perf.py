import io
import numpy as np
import pandas as pd
from sklearn import metrics

trainidx=np.load('/home/yiyou/trainidx.npy')
pred=np.asarray(pd.read_csv('/home/yiyou/test/tmpout/prediction.csv',header=None))
label=np.zeros(252009)
labelidx=np.asarray(pd.read_csv('/home/yiyou/test/label.csv'))
label[labelidx-1]=1
label=label.astype(int)
label=label[trainidx]
pred=pred[trainidx]
predsign=(np.sign(pred)+1)//2
print('train acc: %.3f'%(np.sum(predsign[:,0]==label)/len(label)))
print('train auc: %.3f'%metrics.roc_auc_score(label,pred[:,0]))
print('train mcc: %.3f'%metrics.matthews_corrcoef(label,predsign[:,0]))

pred=np.asarray(pd.read_csv('/home/yiyou/test/tmpout/prediction.csv',header=None))
label=np.zeros(252009)
labelidx=np.asarray(pd.read_csv('/home/yiyou/test/label.csv'))
testidx=np.load('/home/yiyou/testidx.npy')
label[labelidx-1]=1
label=label.astype(int)
label=label[testidx]
pred=pred[testidx]
predsign=(np.sign(pred)+1)//2
print('test acc: %.3f'%(np.sum(predsign[:,0]==label)/len(label)))
print('test auc: %.3f'%metrics.roc_auc_score(label,pred[:,0]))
print('test mcc: %.3f'%metrics.matthews_corrcoef(label,predsign[:,0]))

