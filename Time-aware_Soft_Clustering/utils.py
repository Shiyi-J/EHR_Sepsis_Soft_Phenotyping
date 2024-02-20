import math
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def silhouette_idx(dat_tensor, group_id, template, lambda_rate=0, metric="euclidean", missing="impute"):
    _, num_sub, num_t = dat_tensor.shape
    num_clus = len(template)
    print('K=', num_clus)
    '''
    indices = []
    for i in range(num_sub):
        mat = dat_tensor[:,i,:]
        index = ~np.isnan(mat) # boolean mask
        indices.append(index)
    '''
    if metric == "euclidean":
        # compute a(i), within cluster variance
        ai = np.zeros(num_sub)
        for i in range(num_sub):
            num_cluster = np.sum(group_id == group_id[i])
            cluster_idx = np.where(group_id == group_id[i])[0]
            loss = 0
            for t in range(num_t):
                arr = dat_tensor[:,i,t]
                #i_idx = np.where(indices[i][:,t])[0]
                #if len(i_idx) > 0:
                for k in range(num_cluster):
                    #k_idx = np.where(indices[cluster_idx[k]][:,t])[0]
                    #if len(k_idx) > 0:
                    arr_k = dat_tensor[:,cluster_idx[k],t]
                    #idx = np.logical_and(~np.isnan(arr_k), ~np.isnan(arr))
                    #if np.sum(idx) != 0:
                    loss += distance.euclidean(arr, arr_k)**2
                    '''
                    elif missing == "impute":
                        temp = template[group_id[i]][:,t]
                        arr_imp = arr
                        arr_imp[np.isnan(arr)] = temp[np.isnan(arr)]
                        arr_k[np.isnan(arr_k)] = temp[np.isnan(arr_k)]
                        loss += math.exp(-lambda_rate * t) * distance.euclidean(arr_imp, arr_k)
                    elif missing == "skip":
                        loss += 0
                    '''
            ai[i] = loss / (num_cluster - 1)
        # compute b(i), inter-cluster variance
        bi = np.zeros(num_sub)
        for i in range(num_sub):
            other_cluster = np.unique(group_id[group_id != group_id[i]])
            num_cluster, loss_cl = np.zeros((num_clus,)), np.zeros((num_clus,))
            for cl in other_cluster:
                num_cluster[cl] = np.sum(group_id == cl)
            for t in range(num_t):
                arr = dat_tensor[:,i,t]
                #i_idx = np.where(indices[i][:,t])[0]
                #if len(i_idx) > 0:
                for cl in other_cluster:
                    cluster_idx = np.where(group_id == cl)[0]
                    for k in range(int(num_cluster[cl])):
                        #k_idx = np.where(indices[cluster_idx[k]][:,t])[0]
                        #if len(k_idx) > 0:
                        arr_k = dat_tensor[:,cluster_idx[k],t]
                        #idx = np.logical_and(~np.isnan(arr_k), ~np.isnan(arr))
                        #if np.sum(idx) != 0:
                        loss_cl[cl] += distance.euclidean(arr, arr_k)**2
                        '''
                        else:
                            temp = template[group_id[cl]][:,t]
                            arr_imp = arr
                            arr_imp[np.isnan(arr)] = temp[np.isnan(arr)]
                            arr_k[np.isnan(arr_k)] = temp[np.isnan(arr_k)]
                            loss_cl[cl] += math.exp(-lambda_rate * t) * distance.euclidean(arr_imp, arr_k)
                        '''
            bi[i] = min(loss_cl[other_cluster] / num_cluster[other_cluster])
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters.
    s = (bi - ai) / np.maximum(ai, bi)
    avg = np.mean(s)
    return s, avg

def ALS_reg_completion(matrix, omega, rank, time_range=140, iter_num=5, penalty=0.1):
    # expand the matrix
    n1, n2 = matrix.shape
    matrix_expand = np.zeros((n1, time_range))
    matrix_expand[:, :n2] = matrix.copy()
    omega_raw = omega
    omega = np.zeros((n1, time_range)) == 1
    omega[:, :n2] = omega_raw.copy()

    # diagonal deleteion spectral initialization
    n2 = time_range
    sparsity = np.sum(omega) / np.product(omega.shape)
    gram = matrix_expand.T.dot(matrix_expand) / sparsity
    for i in range(gram.shape[0]):
        gram[i][i] = 0
    V, _, _ = np.linalg.svd(gram)
    V = V[:, :rank]

    # ALS
    for k in range(iter_num):
        U = np.zeros((n1, rank))
        # update U
        for i in range(n1):
            omega_i = omega[i, :]
            if sum(omega_i) == 0:
                continue
            else:
                y = matrix_expand[i, omega_i]
                X = V[omega_i, :]
                U[i, :] = np.linalg.inv(X.T.dot(X) + penalty * np.identity(X.shape[1])).dot(X.T.dot(y))
        # update shift
        Xhat = U.dot(V.T)
        # print(np.sum((Xhat - matrix_expand)**2 * omega))
        for i in range(n1):
            record = matrix_expand[i, omega[i, :]]
            idx = np.where(omega[i, :]==1)[0]
            if len(idx) == 0:
                continue
            idx = idx - idx[0]
            min_loss = np.sum((Xhat[i, idx] - record) ** 2)
            min_idx = idx
            while idx[-1] < time_range-1:
                idx = idx + 1
                cur_loss = np.sum((Xhat[i, idx] - record) ** 2)
                if cur_loss <= min_loss:
                    min_idx, min_loss = idx, cur_loss
            temp, omegai = np.zeros(time_range), np.zeros(time_range) == 1
            temp[min_idx] = record
            matrix_expand[i,:] = temp
            omegai[min_idx] = True
            omega[i,:] = omegai
            # update V
        V = np.zeros((time_range, rank))
        for j in range(time_range):
            omega_j = omega[:, j]
            if sum(omega_j) == 0:
                continue
            else:
                y = matrix_expand[omega_j, j]
                X = U[omega_j, :]
                V[j, :] = np.linalg.inv(X.T.dot(X)+penalty * np.identity(X.shape[1])).dot(X.T.dot(y))
    # find final shift
    Xhat = U.dot(V.T)
    mat_est = np.zeros((n1, n2))
    total_loss = 0
    record_loss = 0
    for i in range(n1):
        idx = np.where(omega_raw[i, :])[0]
        if len(idx) == 0:
            continue
        cur_shift = - idx[0]
        idx = idx - idx[0]
        min_shift = cur_shift
        record = matrix[i, omega_raw[i, :]]
        record_loss += np.sum(record**2)
        min_loss = np.sum((Xhat[i, idx] - record) ** 2)
        while idx[-1] < time_range-1:
            idx = idx + 1
            cur_shift += 1
            cur_loss = np.sum((Xhat[i, idx] - record) ** 2)
            if cur_loss <= min_loss:
                min_shift, min_loss = cur_shift, cur_loss
        total_loss += min_loss
        if min_shift > time_range - n2:
            mat_est[i, :(time_range-min_shift)] = Xhat[i, min_shift:]
        elif min_shift < 0:
            mat_est[i, -(min_shift):] = Xhat[i, :(n2+min_shift)]
        else:
            mat_est[i, :] = Xhat[i, min_shift:(min_shift+n2)]
    return mat_est
