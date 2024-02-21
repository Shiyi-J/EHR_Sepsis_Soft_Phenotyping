import itertools
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def calc_mv_avg(data, w_sz):
    num_fea, num_sub, num_t = data.shape
    new_fea = int(num_t / w_sz)
    final_mat = np.zeros((num_sub, num_fea * new_fea))
    for i in range(num_sub):
        row = []
        for j in range(num_fea):
            for k in range(new_fea):
                row.append(np.nanmean(data[j,i,k*w_sz:(k+1)*w_sz]))
        row = np.nan_to_num(row, nan=np.nanmean(row))
        final_mat[i,:] = row
    return final_mat

def init_temp(dat_tensor, omega, num_fea, curve_range, icd_df, K=4, method='ALS'):
    k_template = []
    tensor_est = np.zeros(dat_tensor.shape)
    dat_tensor_temp = dat_tensor.copy()
    dat_tensor_temp[~omega] = 0
    for t in range(num_fea):
        matrix = dat_tensor_temp[t,:,:].copy()
        if method == 'ALS':
            alpha = np.nanmean(matrix, axis=1)
            beta = np.nanmean(matrix, axis=0)
            alpha[np.isnan(alpha)] = 0
            beta[np.isnan(beta)] = 0
            alpha = np.expand_dims(alpha, axis=1)
            alpha = np.repeat(alpha, curve_range, axis=1)
            matrix = matrix - alpha
            matrix_est = ALS_reg_completion(matrix, omega[t,:,:], rank=11, time_range=curve_range, iter_num=5, penalty=1)
            matrix_est = matrix_est + alpha
        else:
            print('Method does not exist!')
        tensor_est[t,:,:] = matrix_est
    # count obs
    count = np.sum(omega, axis=2)
    count_allfea = np.sum(count, axis=0)
    threshold = 100
    idx = np.where(count_allfea > threshold)[0]
    # filter subjects
    tensor_est_sub = tensor_est[:,idx,:]
    icd_df = icd_df.iloc[idx,:]
    dat_tensor_sub = dat_tensor[:,idx,:]
    # select templates based on the means
    icd_dict = {0:'Hepatic', 1:'Renal', 2:'Pulmonary'}
    for k in range(K):
        mask = (icd_df[icd_dict[k]]==1) & (np.sum(icd_df.iloc[:,1:], axis=1)==1)
        #plot_patients(dat_tensor_sub[:,mask,:], k)
        gr_k = tensor_est_sub[:,mask,:]
        k_template.append(np.mean(gr_k, axis=1))
    torch.save(k_template, 'init_temp_soft_cluster_rm_Jun.pt')
    return k_template, tensor_est
'''
def compute_dist(dat_mat, ind_mat, temp_list, K, num_t):
    # compute distance of single sample to each template
    D_i = np.zeros((K,))
    for t in range(num_t):
        arr = dat_mat[:,t]
        idx = np.where(ind_mat[:,t])[0]
        if len(idx) > 0:
            for k, temp in enumerate(temp_list):
                D_i[k] += distance.euclidean(arr[idx], temp[idx,t])
    return D_i
'''
def compute_tensor_dist_single(dat_mat, features, temp, num_t, cov_tensor=None, typ='Mahalanobis', tol=1e-6):
    # dat_mat contains no missing values
    num_fea = dat_mat.shape[0]
    dist = torch.tensor(0.)
    for j, feat in enumerate(features):
        arr = dat_mat[j,:]
        if typ == 'Euclidean':
            if feat in ['Arterial Blood Pressure systolic','Base Excess','Respiratory Rate']:
                dist += torch.sum((arr[:24] - temp[j,:24])**2) * num_t / 24
            else:
                dist += torch.sum((arr - temp[j,:])**2)
        elif typ == 'Mahalanobis':
            if cov_tensor.shape != (num_t, num_fea, num_fea):
                cov_tensor = cov_tensor.view(num_t, num_fea, num_fea)
            diff = (arr - temp[:,t]).view(-1,1)
            dist += torch.sqrt(torch.t(diff) @ torch.pinverse(cov_tensor[t,:,:], rcond=tol) @ diff + tol)[0,0]
    return dist

def compute_tensor_dist(dat_mat, features, temp_list, K, num_t, cov_tensor=None, typ='Mahalanobis'):
    dist = torch.zeros(K)
    for k in range(K):
        if typ == 'Euclidean':
            dist[k] = compute_tensor_dist_single(dat_mat, features, temp_list[k], num_t, typ=typ)
        elif typ == 'Mahalanobis':
            dist[k] = compute_tensor_dist_single(dat_mat, features, temp_list[k], num_t, cov_tensor=cov_tensor[k,:,:,:], typ=typ)
    return dist

def compute_cov_single(dat_est, mem_list, mem_mat):
    f_num, i_num, t_num = dat_est.shape
    mask = torch.zeros(i_num)
    for cc in mem_list:
        mask = (mask==1) | (mem_mat[:,cc]==1)
    cov = torch.zeros(t_num, f_num, f_num)
    for t in range(t_num):
        cov[t,:,:] = torch.from_numpy(np.cov(dat_est[:,mask,t].numpy()))
    return cov

def compute_cov(dat_est, k_num, mem_mat_old):
    f_num, _, t_num = dat_est.shape
    cov_tensor = torch.zeros(k_num, t_num, f_num, f_num)
    for k in range(k_num):
        for t in range(t_num):
            mask = mem_mat_old[:,k]==1
            cov = torch.from_numpy(np.cov(dat_est[:,mask,t].numpy()))
            cov_tensor[k,t,:,:] = cov
    return cov_tensor

def membership_convert(labels, K):
    # label: list of tensors
    num_sub = len(labels)
    M = torch.zeros((num_sub, K)) # membership matrix
    for i, label in enumerate(labels):
        M[i,label] = 1
    return M

def assign_okm(template, dat_est, features, init=True, membership_old=[], mem_mat_old=None, cov_tensor=None):
    # dat_tensor: tensor
    # indices: list of np bool array
    # template, membership_old: list of tensors
    f_num, i_num, t_num = dat_est.shape
    k_num = len(template)
    membership = []
    for i in range(i_num):
        #if init:
        dist = compute_tensor_dist(dat_est[:,i,:], features, template, k_num, t_num, typ='Euclidean')
        #else:
            #dist = compute_tensor_dist(dat_est[:,i,:], template, k_num, t_num, cov_tensor=cov_tensor, typ='Mahalanobis')
        sort = torch.argsort(dist)
        temp_sort_idx = sort[dist[sort] != 0]
        m = []
        c = temp_sort_idx[0]
        m.append(c)
        phi = template[c] / len(m)

        flag = 0
        for kk in range(1, len(temp_sort_idx)):
            # step 2
            m_prime = m.copy()
            m_prime.append(temp_sort_idx[kk])
            sum = torch.zeros((f_num, t_num))
            for cc in m_prime:
                sum += template[cc]
            phi_prime = sum / len(m_prime)
            # step 3
            #if init:
            dist_prime = compute_tensor_dist_single(dat_est[:,i,:], features, phi_prime, t_num, typ='Euclidean')
            dist_ori = compute_tensor_dist_single(dat_est[:,i,:], features, phi, t_num, typ='Euclidean')
            #else:
                #cov_prime = compute_cov_single(dat_est, m_prime, mem_mat_old)
                #cov_ori = compute_cov_single(dat_est, m, mem_mat_old)
                #dist_prime = compute_tensor_dist_single(dat_est[:,i,:], phi_prime, t_num, cov_tensor=cov_prime, typ='Mahalanobis')
                #dist_ori = compute_tensor_dist_single(dat_est[:,i,:], phi, t_num, cov_tensor=cov_ori, typ='Mahalanobis')
            if dist_prime < dist_ori:
                m = m_prime
                phi = phi_prime
            else:
                if len(membership_old) > 0:
                    sum = torch.zeros((f_num, t_num))
                    for cc in membership_old[i]:
                        sum += template[cc]
                    phi_old = sum / len(membership_old[i])
                    #if init:
                    dist_old = compute_tensor_dist_single(dat_est[:,i,:], features, phi_old, t_num, typ='Euclidean')
                    #else:
                        #cov_old = compute_cov_single(dat_est, membership_old[i], mem_mat_old)
                        #dist_old = compute_tensor_dist_single(dat_est[:,i,:], phi_old, t_num, cov_tensor=cov_old, typ='Mahalanobis')
                    if dist_ori < dist_old:
                        break
                    else:
                        membership.append(membership_old[i])
                        flag = 1
                        break
                else:
                    break
        if flag == 0:
            membership.append(torch.tensor(m))
    return membership

def compute_t_ik(dat_tensor, features, temp_list, num_sub, num_t, num_k, eta):
    dist_mat, u_ike = torch.zeros((num_sub, num_k)), torch.zeros((num_sub, num_k))
    for i in range(num_sub):
        for k, temp in enumerate(temp_list):
            dist = compute_tensor_dist_single(dat_tensor[:,i,:], features, temp, num_t, typ='Euclidean')
            dist_mat[i,k] = dist
        for k in range(num_k):
            mid = (dist_mat[i,k].detach().clone() / dist_mat[i,:].detach().clone()) ** (2 / (eta - 1))
            u_ike[i,k] = (1 / torch.sum(mid)) ** eta
    '''
    gamma, t_ik = torch.zeros((num_k,)), torch.zeros((num_sub, num_k))
    for k in range(num_k):
        gamma[k] = torch.sum(u_ike[:,k] * dist_mat[:,k].detach().clone()) / torch.sum(u_ike[:,k])
        t_ik[:,k] = 1 / (1 + (dist_mat[:,k].detach().clone() / gamma[k]) ** (2 / (eta - 1)))
    '''
    return dist_mat, u_ike #gamma, t_ik

def compute_tot_loss(dat_tensor, features, temp_list, membership, cov_ts, optimizer, icd_df, num_k, b1, b2, eta):
    # b1, b2 for tuning cluster purity
    # eta for tuning fuzzyness
    _, num_sub, num_t = dat_tensor.shape
    # unsupervised loss
    dist_mat, u_ike = compute_t_ik(dat_tensor, features, temp_list, num_sub, num_t, num_k, eta)
    #unsup_loss = torch.sum(t_ik ** eta * dist_mat) + torch.sum(gamma * torch.sum((1 - t_ik) ** eta, dim=0))
    unsup_loss = torch.sum(u_ike * dist_mat)
    # supervised loss
    icd_label = torch.tensor(icd_df.iloc[:,1:].values)
    t_loss, nt_loss = torch.tensor(0.), torch.tensor(0.)
    for i in range(num_sub):
        i_label = icd_label[i]
        if torch.sum(i_label) == 1:
            idx = torch.nonzero(i_label==1).item()
            for k in range(num_k):
                if k == idx:
                    t_loss += dist_mat[i,k]
                else:
                    nt_loss += dist_mat[i,k]
    sup_loss = b1 * t_loss - b2 * nt_loss
    tot_loss = unsup_loss + sup_loss
    tot_loss.backward()
    optimizer.step()
    return temp_list, icd_label, tot_loss.item(), unsup_loss.item(), sup_loss.item(), t_loss.item(), nt_loss.item()
'''
def compute_gini(temp_list, membership, icd_df, num_k, num_sub):
    gini, impurity = torch.ones((num_k,)), 0
    icd_label = torch.tensor(icd_df.iloc[:,1:4].values)
    icd_pure = torch.sum(icd_label, dim=1)==1
    for k in range(num_k):
        n_k = membership[:,k]==1 # mask of all data in cluster k
        for l in range(icd_label.size(1)):
            l_mask = n_k & icd_pure & (icd_label[:,l]==1) # mask of label l data in cluster k
            val = (torch.sum(l_mask).item() / torch.sum(n_k).item())**2
            gini[k] -= val
        impurity += (torch.sum(n_k).item() / num_sub) * gini[k].item()
    print('Current impurity:', impurity)
    return gini
'''
def KHM_OKM_hybrid(dat_recover, features, temp_list, icd_df, beta=1, gamma=1, eta=2, t_max=20, l_rate=1e-4, K=4):
    # Update templates using KHM
    num_fea, num_sub, num_t = dat_recover.shape
    '''
    P = np.zeros((num_sub, K))
    for i in range(num_sub):
        D_i = compute_dist(dat_tensor[:,i,:], indices[i], temp_list, K, num_t)
        d_min_idx = np.argmin(D_i)
        R_i = [D_i[d_min_idx] / d_i_k for d_i_k in D_i]
        pre_div = [r_i_k**2 for k, r_i_k in enumerate(R_i) if k != d_min_idx]
        div = (1 + np.sum(pre_div))**2
        Q_i = [(r_i_k**3 * D_i[d_min_idx]) / div for r_i_k in R_i]
        P[i] = np.array([q_i_k / np.sum(Q_i) for q_i_k in Q_i])
    for k, temp in enumerate(temp_list):
        tmp = np.zeros((num_fea, num_t))
        for i in range(num_sub):
            tmp += P[i,k] * dat_recover[:,i,:]
        temp = tmp
    '''
    # OKM step
    dat_recover = torch.tensor(dat_recover).float()
    temp_list = [nn.Parameter(torch.tensor(temp).float(), requires_grad=True) for temp in temp_list]
    optimizer = optim.SGD(temp_list, lr=l_rate)
    M_list = assign_okm(temp_list, dat_recover, features, init=True)
    M = membership_convert(M_list, K)
    #cov_tensor = compute_cov(dat_recover, K, M)
    t = 0
    tot, unsup, sup, t_l, nt_l = [], [], [], [], []
    while t < t_max:
        T, S = torch.zeros((K, num_fea, num_t)), torch.zeros((K,))
        for k, temp in enumerate(temp_list):
            mask = M[:,k] == 1
            dat_k = dat_recover[:,mask,:]
            _, num_sub_k, num_t_k = dat_k.shape
            for i in range(num_sub_k):
                num_l = torch.sum(M[i])
                alpha_i = torch.tensor(1.) / num_l**2
                c_list = [tmp for l, (tmp, c) in enumerate(zip(temp_list, M[i])) if (c.item()==1.) & (l!=k)]
                if len(c_list) > 0:
                    pre_div = torch.stack(c_list, dim=0)
                    temp_ik = num_l * dat_k[:,i,:] - torch.sum(pre_div, dim=0)
                else:
                    temp_ik = num_l * dat_k[:,i,:]
                T[k] += alpha_i * temp_ik
                S[k] += alpha_i
            temp = T[k] / S[k]
        temp_list, icd_label, tot_l, unsup_l, sup_l, target_l, nontarget_l = compute_tot_loss(dat_recover, features, temp_list, M, None, optimizer, icd_df, K, beta, gamma, eta)
        tot.append(tot_l)
        unsup.append(unsup_l)
        sup.append(sup_l)
        t_l.append(target_l)
        nt_l.append(nontarget_l)
        M_list = assign_okm(temp_list, dat_recover, features, False, M_list, M, None)
        M = membership_convert(M_list, K)
        #cov_tensor = compute_cov(dat_recover, K, M)
        t += 1
    # plot loss
    losses = {'tot':tot, 'unsup':unsup, 'sup':sup, 't_l':t_l, 'nt_l':nt_l}
    plt.figure(figsize=(20,15))
    for p, (k, v) in enumerate(losses.items()):
        plt.subplot(2, 3, p+1)
        plt.plot(v)
        plt.title(k)
    plt.savefig('soft_cluster_loss_plots/Loss_'+str(beta)+'_'+str(gamma)+'_'+str(eta)+'_rm_24hr_Jun.pdf')
    # Compute final M via KHM/PKM
    #dist_mat, gamma, t_ik = compute_t_ik(dat_recover, temp_list, num_sub, num_t, K, eta)
    for i in range(num_sub):
        div = []
        for k, temp in enumerate(temp_list):
            mid = compute_tensor_dist_single(dat_recover[:,i,:], features, temp.detach(), num_t, typ='Euclidean')
            div.append(mid)
            #div.append(1 / mid)
        #div_sum = np.sum(div)
        #M[i] = torch.stack([ele / div_sum for ele in div])
        M[i] = torch.stack(div)
    return [temp.detach().numpy() for temp in temp_list], M.numpy()
