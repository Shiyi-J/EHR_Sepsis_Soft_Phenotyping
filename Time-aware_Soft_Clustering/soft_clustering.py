import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy import interpolate
from scipy.spatial import distance
from scipy.stats import f_oneway
from scipy.io import savemat
import statsmodels.api as sm
import torch
import time
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import *
from soft_cluster_utils import *

def normalize(df, train=True, mu_tmp=None, sigma_tmp=None, full_features=None, if_filter=True):
    features = np.unique(df['feature'])
    if train:
        if if_filter==False:
            featureGroup = df.groupby('feature')
            mu_lst, std_lst = featureGroup.transform('mean').response.values, featureGroup.transform('std').response.values
            mu_tmp = featureGroup['response'].agg(['mean'])['mean'].to_numpy().reshape((1,len(features)))
            sigma_tmp = featureGroup['response'].agg(['std'])['std'].to_numpy().reshape((1,len(features)))
            df.loc[:,'response'] = (df.loc[:,'response'].values - mu_lst) / std_lst
            return df, mu_tmp, sigma_tmp, features
        else:
            featureGroup = df.groupby('feature')
            mu_lst, std_lst = featureGroup.transform('mean').response.values, featureGroup.transform('std').response.values
            mu_tmp = featureGroup['response'].agg(['mean'])['mean'].to_numpy().reshape((1,len(features)))
            sigma_tmp = featureGroup['response'].agg(['std'])['std'].to_numpy().reshape((1,len(features)))
            upperlim = mu_lst + 3 * std_lst
            lowerlim = mu_lst - 3 * std_lst
            lowidx = df[df.loc[:,'response'].values > upperlim].index
            upidx = df[df.loc[:,'response'].values < lowerlim].index
            df = df.drop(upidx|lowidx)
            featureGroup = df.groupby('feature')
            mu_lst, std_lst = featureGroup.transform('mean').response.values, featureGroup.transform('std').response.values
            mu_tmp = featureGroup['response'].agg(['mean'])['mean'].to_numpy().reshape((1,len(features)))
            sigma_tmp = featureGroup['response'].agg(['std'])['std'].to_numpy().reshape((1,len(features)))
            df.loc[:,'response'] = (df.loc[:,'response'].values - mu_lst) / std_lst
            return df, mu_tmp, sigma_tmp, features
    else:
        for name in features:
            assert name in full_features
            f_mu = mu_tmp[0, full_features==name][0]
            f_std = sigma_tmp[0, full_features==name][0]
            df.loc[df['feature']==name,'response'] = (df.loc[df['feature']==name,'response'].values - f_mu) / f_std
        return df, features

def scale(df, subid, sc_dict):
    datlist, new_id = [], []
    for i, id in enumerate(subid):
        temp = df[df['SUBJECT_ID']==id]
        #temp['response'] = temp['response'] / np.maximum(np.ones(temp['time'].shape), (sc_dict[id] - temp['time']))
        if len(temp) > 0:
            datlist.append(temp)
            new_id.append(id)
    return datlist, new_id

def makeTensor(datlist, features, curve_range):
    num_sub = len(datlist)
    dat_tensor, raw_indices = [], []
    for j in features:
        record_j = np.nan * np.zeros((num_sub, curve_range)) # feature slice
        for i in range(num_sub):
            temp = datlist[i][datlist[i]['feature']==j]
            t_step = [int(t) for t in temp['time'].values] # record the t_step with records
            record_j[i, t_step] = temp['response'].values
        dat_tensor.append(record_j)
    dat_tensor = np.concatenate(dat_tensor).reshape(len(features), num_sub, -1) # feature, subject, time
    for i in range(num_sub):
        mat = dat_tensor[:,i,:]
        index = ~np.isnan(mat) # boolean mask
        raw_indices.append(index)
    return dat_tensor, np.array(raw_indices)

def intextplt(seq):
    x_bool = ~np.isnan(seq)
    if sum(x_bool) == 0:
        return seq
    else:
        if np.isnan(seq[0]):
            seq[0] = seq[x_bool][0]
        if np.isnan(seq[-1]):
            seq[-1] = seq[x_bool][-1]
        # update after filling ends
        x_bool = ~np.isnan(seq)
        x = np.arange(len(seq))[x_bool]
        y = seq[x_bool]
        f = interpolate.interp1d(x, y, kind='linear')
        seq = f(np.arange(len(seq)))
        return seq

def MVFR(datlist, features, icd_df, beta, gamma, eta, curve_range=120, t_max=100, temp=None):
    num_sub, num_fea = len(datlist), len(features)
    # convert to tensor format
    dat_tensor, indices = makeTensor(datlist, features, curve_range) # feature, subject, time
    omega = ~np.isnan(dat_tensor)
    temp_list, tensor_est = init_temp(dat_tensor, omega, num_fea, curve_range, icd_df, K=3)
    temp_list, cluster_proba = KHM_OKM_hybrid(tensor_est, features, temp_list, icd_df, beta, gamma, eta, t_max, l_rate=1e-5, K=3)
    return temp_list, cluster_proba

def smooth(x, y, xgrid, lowess):
    sample_idx = np.random.choice(len(x), 50, replace=True)
    y_s = y[sample_idx]
    x_s = x[sample_idx]
    y_sm = lowess(y_s, x_s, frac=1./4., it=20, return_sorted=False)
    # regularly sample it onto the grid
    y_grid = interpolate.interp1d(x_s, y_sm, fill_value='extrapolate')(xgrid)
    return y_grid

def plot_cluster(template, features, beta, gamma, eta):
    lowess = sm.nonparametric.lowess
    num_temp, num_fea = len(template), len(features)
    plt.figure(figsize=(30,28)) #figsize=(30,30)
    plt.rc('axes', titlesize=20) # fontsize of the axes title
    plt.rc('axes', labelsize=20) # fontsize of the x and y labels
    plt.rc('xtick', labelsize=15) # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=20)
    m_dict = {0:('.','red','liver'), 1:('v','orange','kidney'), 2:('s','green','lung'), 3:('P','blue'), 4:('D','purple'), 5:('X','cyan')}
    for j in range(num_fea):
        for k in range(num_temp):
            ax = plt.subplot(num_fea//2+1, 2, j+1)
            val = template[k][j].copy()
            #plt.scatter(np.arange(len(val)), val, marker=m_dict[k][0], c=m_dict[k][1], alpha=0.25)
            #plt.plot(val, c=m_dict[k][1])
            ma = ~np.isnan(val)
            x = np.arange(len(val))[ma]
            xgrid = np.linspace(x.min(),x.max())
            K = 100
            smooths = np.stack([smooth(x, val[ma], xgrid, lowess) for k in range(K)]).T
            mean = np.nanmean(smooths, axis=1)
            c25 = np.nanpercentile(smooths, 2.5, axis=1)
            c97 = np.nanpercentile(smooths, 97.5, axis=1)
            ax.fill_between(xgrid, c25, c97, color=m_dict[k][1], alpha=0.25)
            ax.plot(xgrid, mean, marker=m_dict[k][0], c=m_dict[k][1], label=m_dict[k][2])
            ax.set_xlim(-1, 121)
            ax.set_xlabel('Hours')
            ax.set_yticks(np.linspace(-0.2, 0.4, num=15)) #num=24
            ax.set_ylabel('Normalized values')
            ax.set_title(features[j])
            ax.text(-0.1, 1.1, string.ascii_uppercase[j], transform=ax.transAxes, size=20, weight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
    plt.savefig('Multi-template_per_feature_soft_cluster_'+str(beta)+'_'+str(gamma)+'_'+str(eta)+'_rm_24hr.pdf', bbox_inches='tight')
    return ;

def makeDF(tensor_est, subid, features):
    num_t = tensor_est.shape[-1]
    df_filled = []
    for i, id in enumerate(subid):
        for j, feat in enumerate(features):
            df_feat = pd.DataFrame(columns=['SUBJECT_ID','feature','time','response'])
            df_feat['SUBJECT_ID'] = [id] * num_t
            df_feat['feature'] = [feat] * num_t
            df_feat['time'] = np.arange(num_t)
            df_feat['response'] = tensor_est[j,i,:]
            df_filled.append(df_feat)
    df_filled = pd.concat(df_filled)
    return df_filled

def preprocess(df, sc_dict, curve_range, icd_df, beta, gamma, eta, t_max, label=False, plot=True):
    df = df.dropna(subset=['response'])
    df, mu, sigma, features = normalize(df, train=True, mu_tmp=None, sigma_tmp=None, full_features=None)
    subid = np.unique(df['SUBJECT_ID'])
    datlist, _ = scale(df, subid, sc_dict)

    # test section
    dat_tensor, _ = makeTensor(datlist, features, curve_range)
    omega = ~np.isnan(dat_tensor)
    _, tensor_est = init_temp(dat_tensor, omega, len(features), curve_range, icd_df, K=3)
    df_filled = makeDF(tensor_est, subid, features)

    # return registered data (no registration for this paper)
    register_data = []
    for i, id in enumerate(subid):
        #temp = datlist[i]
        #temp['register_time'] = temp['time'].values + shifts[i]
        #temp['SUBJECT_ID'] = [subid[i]] * len(temp)
        temp = df_filled[df_filled['SUBJECT_ID']==id]
        register_data.append(temp)
    register_data = pd.concat(register_data)
    if label == False:
        template, cluster_mat = MVFR(datlist, features, icd_df, beta, gamma, eta, curve_range=curve_range, t_max=t_max)
        torch.save(template, 'multi-template_soft_cluster_'+str(beta)+'_'+str(gamma)+'_'+str(eta)+'_l2_rm_24hr.pt')
        cluster_id = pd.DataFrame(cluster_mat, columns=['0','1','2'])
        cluster_info = pd.concat((pd.DataFrame(subid, columns=['SUBJECT_ID']), cluster_id), axis=1)
        cluster_info.to_csv('soft_cluster_dist_'+str(beta)+'_'+str(gamma)+'_'+str(eta)+'_l2_rm_24hr.csv', index=False)
        if plot:
            plot_cluster(template, features, beta, gamma, eta)
        return register_data, template, features, mu, sigma
    else:
        return register_data, None, None, None, None
