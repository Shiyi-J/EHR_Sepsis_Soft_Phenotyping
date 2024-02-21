import sys
import math
import random
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from Classification_RNN import *
from Baselines import *
from utils import *
from soft_clustering import *
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def compute_auc(y_te, y_proba, length, plot=False):
    n_class = np.unique(y_te)
    auroc, auprc = [], []
    plt.figure(figsize=(10,8))
    for i in n_class:
        y_true = y_te==i
        fpr, tpr, thres = metrics.roc_curve(y_true, y_proba[:,int(i)])
        pre, rec, thresholds = precision_recall_curve(y_true, y_proba[:,int(i)])
        auroc.append(metrics.auc(tpr, fpr))
        auprc.append(metrics.auc(rec, pre))
        plt.plot(rec, pre, label='Class'+str(int(i)))
    if plot:
        plt.legend(loc='best')
        plt.savefig('sepsis_auprc_t='+str(length)+'_soft_cluster.pdf')
    return np.mean(auroc), np.mean(auprc)

def compute_feature(reg, df_lb, length, time, df_info):
    fea_name = np.unique(reg['feature'])
    df = pd.DataFrame()
    for name in fea_name:
        val = reg[reg['feature']==name]['response'].values
        val = np.nan_to_num(val)
        reg.loc[reg['feature']==name, 'response'] = val
        fea_data = gen_subj_data(reg, name, df_lb['SUBJECT_ID'].values, length, time)
        df = pd.concat([df, pd.DataFrame(fea_data)], axis=1)
    # add demographic features
    add = []
    for subj in df_lb['SUBJECT_ID'].values:
        age = df_info[df_info['subject_id']==subj]['anchor_age'].values[0]
        gender = df_info[df_info['subject_id']==subj]['gender'].values[0]
        if gender=='M':
            add.append([age, 1])
        elif gender=='F':
            add.append([age, 0])
        else:
            print('Something went wrong!')
    df = pd.concat([df, pd.DataFrame(add)], axis=1)
    return df.values, df_lb['cluster'].values-1

def test_prediction(df, lb, df_info, icd_df, subj_id, scale_dict, curve_range, te_sz, method, n_repeat, lmbd, beta, gamma, K, t_max, length):
    log_auc, log_pre, log_rec, log_acc, log_auprc = [], [], [], [], []
    for i in range(n_repeat):
        if lb == False: # do not have labels generated from post-soft clustering analysis
            X_reg, _, _, _, _ = preprocess(df, scale_dict, curve_range, icd_df, beta, gamma, lmbd, t_max, lb, plot=True)
            sys.exit()
        else:
            X_reg, _, _, _, _ = preprocess(df, scale_dict, curve_range, icd_df, beta, gamma, lmbd, t_max, lb, plot=False)
            df_lb = pd.read_csv('sepsis_soft_cluster_hard_label_iv.csv')
        # start prediction
        X_reg['time'] = X_reg['time'].astype(int)
        tr_subj, te_subj = train_test_split(subj_id, test_size=te_sz, random_state=42)
        X_train, X_test = X_reg[X_reg['SUBJECT_ID'].isin(tr_subj)], X_reg[X_reg['SUBJECT_ID'].isin(te_subj)]
        y_train, y_test = df_lb[df_lb['SUBJECT_ID'].isin(tr_subj)], df_lb[df_lb['SUBJECT_ID'].isin(te_subj)]
        X_tr, y_tr = compute_feature(X_train, y_train, length, 'time', df_info)
        X_te, y_te = compute_feature(X_test, y_test, length, 'time', df_info)
        if method == 'LR':
            clf = LogisticRegression().fit(X_tr, y_tr)
            y_proba = clf.predict_proba(X_te)
            #auc = roc_auc_score(y_te, y_proba, multi_class='ovo')
            y_hat = np.argmax(y_proba, axis=1)
            pre = metrics.precision_score(y_te, y_hat, average='macro')
            rec = metrics.recall_score(y_te, y_hat, average='macro')
            acc = metrics.accuracy_score(y_te, y_hat)
            auroc, auprc = compute_auc(y_te, y_proba, length, plot=True)
            torch.save(clf, 'sepsis_predict_mdl_'+str(length)+'hr_add.pt')
            cm = metrics.plot_confusion_matrix(clf, X_te, y_te, normalize='true')
            cm.figure_.savefig('cmat_'+str(length)+'hr_add.pdf')
        else:
            print('Method not defined yet!')
        log_auc.append(auroc)
        log_pre.append(pre)
        log_rec.append(rec)
        log_acc.append(acc)
        log_auprc.append(auprc)
    return [log_auc, log_pre, log_rec, log_acc, log_auprc]
