import numpy as np
import pandas as pd
from sklearn import metrics

def calculate_metrics(pred_path, label_path, idx_path, train=True):
    # load data
    pred = np.asarray(pd.read_csv(pred_path, header=None))
    label = np.zeros(252009)
    labelidx = np.asarray(pd.read_csv(label_path))
    
    # adjust labels to binary (1 or 0)
    label[labelidx - 1] = 1
    label = label.astype(int)

    #load the corresponding indices (train or test)
    if train:
        idx = np.load(idx_path)  # Train indices
    else:
        idx = np.load(idx_path)  # Test indices
    
    # filter labels and predictions based on indices
    label = label[idx]
    pred = pred[idx]
    
    # Sign transformation: (np.sign(pred) + 1) // 2 to convert to binary (0 or 1)
    predsign = (np.sign(pred) + 1) // 2

    # Print metrics: accuracy, AUC, MCC
    acc = np.sum(predsign[:, 0] == label) / len(label)
    auc = metrics.roc_auc_score(label, pred[:, 0])
    mcc = metrics.matthews_corrcoef(label, predsign[:, 0])

    return acc, auc, mcc


train_pred_path = './train/prediction.csv'
train_label_path = './train/label.csv'
train_idx_path = './train/trainidx.npy'

test_pred_path = '/test/prediction.csv'
test_label_path = './test/label.csv'
test_idx_path = './test/testidx.npy'


# Train metrics
train_acc, train_auc, train_mcc = calculate_metrics(train_pred_path, train_label_path, train_idx_path, train=True)
print(f'train acc: {train_acc:.3f}')
print(f'train auc: {train_auc:.3f}')
print(f'train mcc: {train_mcc:.3f}')

# Test metrics
test_acc, test_auc, test_mcc = calculate_metrics(test_pred_path, test_label_path, test_idx_path, train=False)
print(f'test acc: {test_acc:.3f}')
print(f'test auc: {test_auc:.3f}')
print(f'test mcc: {test_mcc:.3f}')
