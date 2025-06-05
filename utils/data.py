import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
from nilearn.connectome import sym_matrix_to_vec
from scipy.stats import zscore

import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import pickle
from typing import Tuple, List, Optional, Callable
from nilearn.connectome import ConnectivityMeasure
import numpy as np
    
    
    
class AnyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    

def get_split_ts(strategy, atlas, GSR):
    ihb_closed, ihb_opened, _ = load_ihb(strategy, atlas, GSR)
    china_closed, china_opened, china_y = load_china(strategy, atlas, GSR)

    train_idx, test_idx = train_test_split(
        np.arange(len(ihb_closed)), train_size=0.1, random_state=42)
    
    x_train = np.concatenate([ihb_closed[train_idx], ihb_opened[train_idx], 
                              china_closed[:, :120, :], china_opened[:, :120, :]], axis=0)
    
    y_train = np.concatenate([[0] * len(train_idx), [1] * len(train_idx), 
                              china_y], axis=0)
    
    x_test = np.concatenate([ihb_closed[test_idx], ihb_opened[test_idx]], axis=0)
    
    y_test = np.concatenate([[0] * len(test_idx), [1] * len(test_idx)], axis=0)
    
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = get_split_ts(4, 'AAL', True)


def get_split_fc(fc, atlas, strategy, gsr,
              random_state: int = 42):
    
    """
    Loads functional connectivity data, splits it into training and testing sets,
    and returns vectorized connectivity matrices and labels.

    Parameters:
    - fc (str): Type of functional connectivity ('glasso', 'partial_corr', 'tangent', etc.).
    - atlas (str): Name of the brain atlas used.
    - strategy (str): Strategy used for preprocessing the data.
    - gsr (bool): Whether global signal regression was applied.
    - random_state (int, optional): Seed for random operations. Default is 42.

    Returns:
    - x_train (np.ndarray): Vectorized training connectivity matrices.
    - y_train (np.ndarray): Training labels.
    - x_test (np.ndarray): Vectorized testing connectivity matrices.
    - y_test (np.ndarray): Testing labels.
    """

    if fc != 'glasso':
        ihb_op = np.load(f'/data/Projects/OpenCloseBenchmark_data/fc_data/op/ihb_open_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy')
        ihb_cl = np.load(f'/data/Projects/OpenCloseBenchmark_data/fc_data/cl/ihb_close_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy')

        if fc == 'partial_corr':
            fc = 'pc'
        elif fc == 'tangent':
            fc = 'tang'

    else:
        a = np.load(f'/data/Projects/OpenCloseBenchmark_data/glasso_output/ihb/ihb_{atlas}_{strategy}_{gsr}.npy')
        ihb_op = a[84:]
        ihb_cl = a[:84]

    
    train_groups, test_groups = train_test_split(np.arange(84),
                                                test_size=0.9,
                                                random_state=random_state)


    op_test2 = np.load(f'/data/Projects/OpenCloseBenchmark_data/fc_data_china/open/china_open2_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy')
    op_test3 = np.load(f'/data/Projects/OpenCloseBenchmark_data/fc_data_china/open/china_open3_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy')
    cl_test = np.load(f'/data/Projects/OpenCloseBenchmark_data/fc_data_china/close/china_close1_{fc}_{atlas}_strategy-{strategy}_{gsr}.npy')
    
    x_train = np.concatenate([
        cl_test, ihb_cl[train_groups],
        ihb_op[train_groups], op_test2, op_test3])
    x_train = sym_matrix_to_vec(x_train, discard_diagonal=True)
    

    x_test = np.concatenate([
        ihb_cl[test_groups],
        ihb_op[test_groups]])
    x_test = sym_matrix_to_vec(x_test, discard_diagonal=True)
        
    y_train = np.array(
        [0] * len(cl_test) + [0] * len(train_groups) + [1] * len(train_groups) + [0] * len(op_test2) + [1] * len(op_test3))
    
    y_test = np.array([0] * len(test_groups) + [1] * len(test_groups))

    
    return x_train, y_train, x_test, y_test





def load_rmet(strategy, atlas, GSR):

    sub = [f'{i+1:03d}' for i in range(64) if i!= 52]
    closed = fetch_ts('/data/Projects/OpenCloseChina/ts', sub=sub, GSR=GSR,
                      run=1, task='rest', strategy=strategy, atlas_name=atlas)
    opened = fetch_ts('/data/Projects/OpenCloseChina/ts', sub=sub, GSR=GSR,
                      run=2, task='rest', strategy=strategy, atlas_name=atlas)
    
    if atlas == 'HCPex':
        hcp = pd.read_excel('/home/tm/projects/OpenCloseBaseline/HCPex_Atlas_Description.xlsx',
                            index_col='HCPex_ID')
        
        to_del = [390]
        hcp.drop(to_del, axis=0, inplace=True)
        drop = [365, 372, 396, 398, 401, 405]

        for i in closed:
            i.columns = hcp.index
            i.drop(drop, axis=1, inplace=True)

        for i in opened:
            i.columns = hcp.index
            i.drop(drop, axis=1, inplace=True)
    
    closed = zscore(closed, axis=1, nan_policy='omit')
    opened = zscore(opened, axis=1, nan_policy='omit')

    #np.nan_to_num(closed, copy=False)
    #np.nan_to_num(opened, copy=False)

    n_closed, n_opened = closed.shape[0], opened.shape[0]

    #X = np.concatenate([closed, opened], axis=0)
    y = np.array([0] * n_closed + [1] * n_opened)
    groups = np.array([i for i in range(n_closed)] + 
                      [i for i in range(n_opened)])
    
    return closed, opened, y, groups

def load_ihb(strategy, atlas, GSR):

    coverage = np.load(f'/home/tm/projects/OpenCloseProject/coverage/ihb_{atlas}_parcel_coverage.npy') > 0.1

    sub = [f'{i+1:03d}' for i in range(84)]
    closed = fetch_ts('/data/Projects/OpenCloseIHB/outputs', sub=sub, GSR=GSR,
                      run=1, task='rest', strategy=strategy, atlas_name=atlas)
    opened = fetch_ts('/data/Projects/OpenCloseIHB/outputs', sub=sub, GSR=GSR,
                      run=2, task='rest', strategy=strategy, atlas_name=atlas)
    hcp = None
    if atlas == 'HCPex':
        hcp = pd.read_excel('/home/tm/projects/OpenCloseBaseline/atlas/HCPex_Atlas_Description.xlsx',
                            index_col='HCPex_ID')
        
        to_del = [365, 398, 401]
        hcp.drop(to_del, axis=0, inplace=True)
        drop = [372, 390, 396, 405]

        for i in closed:
            i.columns = hcp.index
            i.drop(drop, axis=1, inplace=True)

        for i in opened:
            i.columns = hcp.index
            i.drop(drop, axis=1, inplace=True)

        coverage = np.delete(coverage, to_del+drop)

    # shape = [sub, ts, roi]

    closed = np.array(closed)[:, :, coverage]
    opened = np.array(opened)[:, :, coverage]

    if atlas == 'HCPex':
        closed = np.delete(closed, [337, 343, 344, 366], axis=2)
        opened = np.delete(opened, [337, 343, 344, 366], axis=2)


    closed = zscore(closed, axis=1, nan_policy='omit') 
    opened = zscore(opened, axis=1, nan_policy='omit')

    n_closed, n_opened = closed.shape[0], opened.shape[0]

    #X = np.concatenate([closed, opened], axis=0)
    y = np.array([0] * n_closed + [1] * n_opened)
    groups = np.array([i for i in range(n_closed)] + 
                      [i for i in range(n_opened)])
    
    return closed, opened, y


def load_china(strategy, atlas, GSR):

    coverage = np.load(f'/home/tm/projects/OpenCloseProject/coverage/ihb_{atlas}_parcel_coverage.npy') > 0.1

    df = pd.read_csv(r"/arch/OpenCloseBeijin/BeijingEOEC.csv")

    open_ids2 = df['SubjectID'].loc[df['Session_2'] == 'open'].values
    open_ids3 = df['SubjectID'].loc[df['Session_3'] == 'open'].values

    closed_ids1 = df['SubjectID'].loc[df['Session_1'] == 'closed'].values
    closed_ids2 = df['SubjectID'].loc[df['Session_2'] == 'closed'].values
    closed_ids3 = df['SubjectID'].loc[df['Session_3'] == 'closed'].values

    closed1 = fetch_ts('/data/Projects/OpenCloseChina/outputs_china',
                  sub=closed_ids1, run=1, GSR=GSR,
                  atlas_name=atlas, strategy=strategy)

    closed2 = fetch_ts('/data/Projects/OpenCloseChina/outputs_china',
                        sub=closed_ids2, run=2, GSR=GSR,
                        atlas_name=atlas, strategy=strategy)

    closed3 = fetch_ts('/data/Projects/OpenCloseChina/outputs_china',
                        sub=closed_ids3, run=3, GSR=GSR,
                        atlas_name=atlas, strategy=strategy)

    opened2 = fetch_ts('/data/Projects/OpenCloseChina/outputs_china',
                        sub=open_ids2, run=2, GSR=GSR,
                        atlas_name=atlas, strategy=strategy)
    
    opened3 = fetch_ts('/data/Projects/OpenCloseChina/outputs_china',
                        sub=open_ids3, run=3, GSR=GSR,
                        atlas_name=atlas, strategy=strategy)
    
    

    for en, i in enumerate(closed3):
        if len(i) < 240:
            np.delete(closed_ids3, en)
            del closed3[en]
    for en, i in enumerate(opened2):
        if len(i) < 240:
            np.delete(open_ids2, en)
            del opened2[en]

    closed = closed1 + closed2 + closed3
    opened = opened2 + opened3

    #del opened[22]
    #del closed[84]

    if atlas == 'HCPex':
        hcp = pd.read_excel('/home/tm/projects/OpenCloseBaseline/atlas/HCPex_Atlas_Description.xlsx',
                            index_col='HCPex_ID')
        
        to_del = [365, 372, 396, 401, 405]
        hcp.drop(to_del, axis=0, inplace=True)
        drop = [390, 398]

        for i in closed:
            i.columns = hcp.index
            i.drop(drop, axis=1, inplace=True)
            
        for i in opened:
            i.columns = hcp.index
            i.drop(drop, axis=1, inplace=True)

        coverage = np.delete(coverage, to_del+drop)
            
    closed = np.array(closed)[:, :, coverage]
    opened = np.array(opened)[:, :, coverage]

    if atlas == 'HCPex':
        closed = np.delete(closed, [337, 343, 344, 366], axis=2)
        opened = np.delete(opened, [337, 343, 344, 366], axis=2)

    n_opened = opened.shape[0]

    closed = zscore(closed, axis=1, nan_policy='raise')[:n_opened]
    opened = zscore(opened, axis=1, nan_policy='raise')


    

    #op, cl = [], []
    #for en, i in enumerate(closed_ids1):
        #cl.append(en)
        #if i in open_ids2 or i in open_ids3:
            #op.append(en)
        #if i in closed_ids2 or i in closed_ids3:
            #cl.append(en)

    #del op[22]
    #del cl[84]

    #X = np.concatenate([closed, opened], axis=0)
    y = np.array([0] * n_opened + [1] * n_opened)
    #groups = np.array(cl + op)

    return closed, opened, y#, groups


def fetch_ts(path, sub=None, run=1, task='rest', strategy=4, atlas_name='AAL', GSR=False):
    ts = []
    if sub is None:
        sub = os.listdir(path)
    failed = []
    for i in sub:
        if not isinstance(i, str):
            i = str(i)
        if 'sub' in i:
            i = i[4:]
        try:
            name = (f'sub-{i}_task-{task}_run-{run}_time-series_{atlas_name}_strategy-{strategy}_GSR.csv' if GSR 
                    else f'sub-{i}_task-{task}_run-{run}_time-series_{atlas_name}_strategy-{strategy}.csv')
            path_to_file = os.path.join(path, f'sub-{i}', 'time-series', atlas_name, name)
            #print(path_to_file)
            ts.append(pd.read_csv(path_to_file))#.values)
        except FileNotFoundError:
            failed.append(i)
            continue

    if len(failed) > 0:
        print('no files available:', failed)
        
    return ts