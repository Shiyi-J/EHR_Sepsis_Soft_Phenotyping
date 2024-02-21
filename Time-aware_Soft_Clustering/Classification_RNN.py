import warnings
warnings.filterwarnings("ignore")
import random
import math
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from scipy import interpolate
from scipy import linalg
import matplotlib.pyplot as plt
from textwrap import wrap

import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim

# helper functions:
def format_data(data, id, feature, length, time):
    # data has been normalized
    subj_data = data[(data['SUBJECT_ID']==id) & (data['feature']==feature)]
    hour = subj_data[time]
    value = subj_data['response']
    hour_value = [(t, num) for t, num in sorted(zip(hour, value)) if t < length]
    new_val = np.zeros((length,))
    if len(hour_value) == 0:
        return new_val, hour_value
    else:
        j = 0
        for i in range(len(new_val)):
            if j < len(hour_value):
                if i == hour_value[j][0] and not np.isnan(hour_value[j][0]):
                    new_val[i] = hour_value[j][1]
                    j += 1
                else:
                    continue
        return new_val, hour_value

def interplt(hour_value, length):
    new_val = np.zeros((length,))
    if len(hour_value) == 0:
        return new_val
    else:
        hour_value = list(zip(*hour_value))
        hour, value = list(hour_value[0]), list(hour_value[1])
        if len(value) == 1:
            new_val[:hour[0]+1] = value[0]
            return new_val
        else: # at least 2 points
            if hour[0] != 0:
                new_val[:hour[0]] = value[0]
                f = interpolate.interp1d(hour, value, kind='linear')
                new_val[hour[0]:hour[-1]+1] = f(np.arange(hour[0], hour[-1]+1))
                new_val = np.nan_to_num(new_val)
            else:
                f = interpolate.interp1d(hour, value, kind='linear')
                new_val[:hour[-1]+1] = f(np.arange(0, hour[-1]+1))
                new_val = np.nan_to_num(new_val)
            return new_val

def gen_gr_data(data, name, subj_gr, length, time, lst, typ):
    if typ == 'interpolation':
        for id in subj_gr:
            d, hr, val = format_data(data, id, name, length, time)
            interp_d = interplt(hr, val, length)
            lst.append(interp_d)
    else:
        for id in subj_gr:
            d, hr, val = format_data(data, id, name, length, time)
            lst.append(d)
    return lst

def prepare_data(data, lb, batch_sz, te_sz):
    data, lb = torch.Tensor(data).float(), torch.Tensor(lb).float()
    X_tr, X_te, y_tr, y_te = train_test_split(data, lb, test_size=te_sz)
    train = torch.utils.data.TensorDataset(X_tr, y_tr.long())
    test = torch.utils.data.TensorDataset(X_te, y_te.long())
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_sz, shuffle=True, num_workers=0, worker_init_fn=random.seed(1))
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_sz, shuffle=False, num_workers=0, worker_init_fn=random.seed(1))
    return train_loader, test_loader

def prepare_data_v2(X_tr, X_te, y_tr, y_te, batch_sz):
    X_tr, X_te = torch.Tensor(X_tr).float(), torch.Tensor(X_te).float()
    y_tr, y_te = torch.Tensor(y_tr).long(), torch.Tensor(y_te).long()
    train = torch.utils.data.TensorDataset(X_tr, y_tr)
    test = torch.utils.data.TensorDataset(X_te, y_te)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_sz, shuffle=True, num_workers=0, worker_init_fn=random.seed(1))
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_sz, shuffle=False, num_workers=0, worker_init_fn=random.seed(1))
    return train_loader, test_loader

def prepare_loader(length, time, typ, reg, label, fea_name):
    processed_d = []
    # filter label here!
    #label = raw_label[raw_label['SUBJECT_ID'].isin(reg['SUBJECT_ID'])]
    subj0 = label[label['DEATH_HOUR']==np.inf]['SUBJECT_ID'].values
    subj1 = label[label['DEATH_HOUR']!=np.inf]['SUBJECT_ID'].values
    for name in fea_name:
        val = reg[reg['feature']==name]['response'].values
        val = np.nan_to_num(val)
        reg.loc[reg['feature']==name, 'response'] = val
        fea_data = []
        fea_data = gen_gr_data(reg, name, subj0, length, time, fea_data, typ)
        fea_data = gen_gr_data(reg, name, subj1, length, time, fea_data, typ)
        processed_d.append(fea_data)
    processed_d = np.transpose(np.array(processed_d), (1,2,0))
    lb = np.concatenate((np.zeros((len(subj0),)), np.ones((len(subj1),))))
    #tr_loader, te_loader = prepare_data(processed_d, lb, batch_sz, te_sz)
    return None, None, processed_d.shape[-1], processed_d, lb

# model:
class GRUModel(nn.Module):
    def __init__(self, num_l, in_sz, hidden_sz, out_sz):
        super(GRUModel, self).__init__()
        self.num_layers = num_l
        self.hid_sz = hidden_sz
        self.gru = nn.GRU(input_size=in_sz, hidden_size=hidden_sz, num_layers=num_l, batch_first=True)
        self.fc = nn.Linear(hidden_sz, out_sz)

    def init_hidden(self, b_sz):
        return torch.zeros(self.num_layers, b_sz, self.hid_sz)

    def forward(self, x):
        b_sz = x.size(0)
        h_state = self.init_hidden(b_sz)
        out, h_state = self.gru(x, h_state.to(device))
        out = self.fc(out[:,-1,:])
        return out

def train_model(epoch, tr_loader, te_loader, model, criterion, optimizer, device, thres):
    log_loss_tr, log_loss_te = [], []
    for e in range(epoch):
        tr_l, te_l = [], []
        model.train()
        for i, (tr_d, tr_lb) in enumerate(tr_loader):
            tr_d, tr_lb = tr_d.to(device), tr_lb.to(device)
            out = model(tr_d)
            loss = criterion(out, tr_lb)
            tr_l.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log_loss_tr.append(np.mean(tr_l))

        model.eval()
        y, yh_proba, yh = [], [], []
        for i, (te_d, te_lb) in enumerate(te_loader):
            te_d, te_lb = te_d.to(device), te_lb.to(device)
            out = model(te_d)
            y_hat = F.softmax(out, dim=1)[:,1].detach().cpu()
            y.append(te_lb.cpu())
            yh_proba.append(y_hat)
            yh.append(1*y_hat>=thres)
            #yh.append(torch.argmax(out, dim=1).cpu())
            loss = criterion(out, te_lb)
            te_l.append(loss.item())
        log_loss_te.append(np.mean(te_l))
        auc = roc_auc_score(torch.cat(y), torch.cat(yh_proba))
        precision, recall, thresholds = precision_recall_curve(torch.cat(y), torch.cat(yh_proba))
        auprc = metrics.auc(recall, precision)
        diff = [abs(pre-rec) for pre, rec in zip(precision, recall)]
        minprc = min(precision[diff.index(min(diff))], recall[diff.index(min(diff))])
        #acc = metrics.accuracy_score(torch.cat(y), torch.cat(yh))
        #pre = metrics.precision_score(torch.cat(y), torch.cat(yh))
        #rec = metrics.recall_score(torch.cat(y), torch.cat(yh))
    return auc, auprc, minprc, precision, recall, thresholds, log_loss_tr, log_loss_te

def gen_results(df, lb, n_repeat, typ, method): #length, time, it
    log_auc, log_auprc, log_minprc, log_pre, log_rec, log_cut = [], [], [], [], [], []
    aft_auc, aft_auprc, aft_minprc, aft_pre, aft_rec, aft_cut = [], [], [], [], [], []
    if method == 'repeat': # old version, use k-fold method
        for i in range(n_repeat):
            tr_loader, te_loader, fea_dim, _, _ = prepare_loader(length, time, typ)
            model = GRUModel(n_layer, fea_dim, hidden_sz, n_class)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=l_rate)
            auc, auprc, minprc, acc, pre, rec, tr_loss, te_loss = train_model(it, tr_loader, te_loader, model, criterion, optimizer, device, thres)
            log_acc.append(acc)
            log_pre.append(pre)
            log_rec.append(rec)
            plt.figure()
            plt.plot(tr_loss, label='Train')
            plt.plot(te_loss, label='Test')
            plt.legend()
            plt.savefig(str(length)+'_loss.pdf')
    elif method == 'k-fold':
        kf = KFold(n_splits=n_repeat, shuffle=True)
        #_, _, fea_dim, processed_d, lb = prepare_loader(length, time, typ)
        subid = np.unique(df['SUBJECT_ID'].values)
        for tr_ind, te_ind in kf.split(subid):
            X_train, X_test = df[df['SUBJECT_ID'].isin(subid[tr_ind])], df[df['SUBJECT_ID'].isin(subid[te_ind])]
            y_train, y_test = lb[lb['SUBJECT_ID'].isin(subid[tr_ind])], lb[lb['SUBJECT_ID'].isin(subid[te_ind])]

            X_train_reg, template, features, mu_tmp, sigma_tmp = preprocess(X_train, scale_dict)
            X_test_reg = fit_template(X_test, template, features, mu_tmp, sigma_tmp, scale_dict)

            X_train_reg['time'] = X_train_reg['time'].astype(int)
            X_train_reg['register_time'] = X_train_reg['register_time'].astype(int)
            X_test_reg['time'] = X_test_reg['time'].astype(int)
            X_test_reg['register_time'] = X_test_reg['register_time'].astype(int)

            features = np.array(['pH','PTT','Base Excess','pO2','Bicarbonate'])
            for length, time, it in zip([120,140], ['time','register_time'], [40,20]):
                _, _, fea_dim_tr, X_tr, y_tr = prepare_loader(length, time, typ, X_train_reg, y_train, features)
                _, _, fea_dim_te, X_te, y_te = prepare_loader(length, time, typ, X_test_reg, y_test, features)
                assert fea_dim_tr == fea_dim_te
                tr_loader, te_loader = prepare_data_v2(X_tr, X_te, y_tr, y_te, batch_sz)
                model = GRUModel(n_layer, fea_dim_tr, hidden_sz, n_class)
                model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=l_rate)
                auc, auprc, minprc, pre, rec, cut, tr_loss, te_loss = train_model(it, tr_loader, te_loader, model, criterion, optimizer, device, thres)
                if length == 120:
                    log_auc.append(auc)
                    log_auprc.append(auprc)
                    log_minprc.append(minprc)
                    log_pre.append(pre)
                    log_rec.append(rec)
                    log_cut.append(cut)
                else:
                    aft_auc.append(auc)
                    aft_auprc.append(auprc)
                    aft_minprc.append(minprc)
                    aft_pre.append(pre)
                    aft_rec.append(rec)
                    aft_cut.append(cut)
            torch.save([log_auc, log_auprc, log_minprc, log_pre, log_rec, log_cut], 'b4_int_rnn_thres_mean_tmpfit_norm_scale_repeat.pt')
            torch.save([aft_auc, aft_auprc, aft_minprc, aft_pre, aft_rec, aft_cut], 'aft_int_rnn_thres_mean_tmpfit_norm_scale_repeat.pt')
    else:
        print('Something went wrong!')
    return [log_auc, log_auprc, log_minprc, log_pre, log_rec, log_cut], [aft_auc, aft_auprc, aft_minprc, aft_pre, aft_rec, aft_cut]

def get_val(lst, method):
    arr = np.zeros((3,3))
    if method == 'formula':
        for i in range(3):
            mu = np.mean(lst[i])
            sigma = np.std(lst[i])
            e_dn = np.percentile(lst[i], 2.5) # ignore this value
            arr[:,i] = np.array([mu, mu-e_dn, sigma])
    else:
        for i in range(3):
            mu = np.mean(lst[i])
            e_up = np.percentile(lst[i], 97.5)
            e_dn = np.percentile(lst[i], 2.5)
            arr[:,i] = np.array([mu, mu-e_dn, e_up-mu])
    return arr

def plot_bar(b4_int, aft_int, n_repeat):
    data = np.array([get_val(b4_int,'formula'), get_val(aft_int,'formula')])
    fill = ['Before registration, interpolation', 'After registration, interpolation']
    fill = ['\n'.join(wrap(l, 20)) for l in fill]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(np.arange(2)-0.1, data[:,0,0], yerr=1.96*data[:,2,0]/math.sqrt(n_repeat), width=0.1, label='AUROC with 95% CI', capsize=3) #data[:,1:,0].T
    ax.bar(np.arange(2), data[:,0,1], yerr=1.96*data[:,2,1]/math.sqrt(n_repeat), width=0.1, label='AUPRC with 95% CI', capsize=3)
    ax.bar(np.arange(2)+0.1, data[:,0,2], yerr=1.96*data[:,2,2]/math.sqrt(n_repeat), width=0.1, label='Min(Re,P+) with 95% CI', capsize=3)
    #ax.bar(np.arange(2)+0.15, data[:,0,3], yerr=1.96*data[:,2,3]/math.sqrt(n_repeat), width=0.1, label='Accuracy with 95% CI', capsize=3) #data[:,1:,1].T
    #ax.bar(np.arange(2)+0.05, data[:,0,2], yerr=1.96*data[:,2,2]/math.sqrt(n_repeat), width=0.1, label='Precision with 95% CI', capsize=3) #data[:,1:,2].T
    #ax.bar(np.arange(2)+0.15, data[:,0,3], yerr=1.96*data[:,2,3]/math.sqrt(n_repeat), width=0.1, label='Recall with 95% CI', capsize=3) #data[:,1:,3].T
    plt.ylim(0,1.1)
    plt.legend(loc='lower left')
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(fill)
    plt.tight_layout()
    plt.savefig('classification_mean_tmpfit_norm_scale_repeat.pdf')
