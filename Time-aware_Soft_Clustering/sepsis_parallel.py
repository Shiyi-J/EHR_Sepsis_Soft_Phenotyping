import pandas as pd
import numpy as np
from sepsis_predict import *
import itertools
from multiprocessing import Pool

curve_range = 120
n_repeat = 1
te_sz = 0.3
raw_data = pd.read_csv('lab_drg_870_872_iv.csv')
raw_data = raw_data.rename(columns={'FEATURE_NAME':'feature', 'RECORD_MIN':'time', 'VALUE':'response'})
raw_data = raw_data[['SUBJECT_ID','feature','time','response']]
raw_data['time'] = round(raw_data['time']) / 60
feat_list = ['Arterial Blood Pressure systolic','Base Excess','Creatinine','INR(PT)',
             'Lactate','Heart Rate','Respiratory Rate']
raw_data = raw_data[raw_data['feature'].isin(feat_list)]
icd_df = pd.read_csv('soft_cluster_icd_label.csv')
icd_label = icd_df.iloc[:,:-1]
subid = np.unique(icd_label[icd_df.iloc[:,-1]==0]['SUBJECT_ID'])
raw_data = raw_data[raw_data['SUBJECT_ID'].isin(subid)]
# filter data
#subid = np.unique(raw_data['SUBJECT_ID'])
data, scale_dict = [], dict()
for i, id in enumerate(subid):
    temp = raw_data[raw_data['SUBJECT_ID']==id]
    scale_dict[id] = max(temp['time'])
    data.append(temp[temp['time'] < curve_range])
data = pd.concat(data)
df_info = pd.read_csv('patients.csv')

def wrap_fcn(args):
    #b, g = args
    # generate labels of hybrid sepsis sub-phenotypes
    _ = test_prediction(data, False, df_info, icd_label, subid, scale_dict, curve_range, te_sz, 'LR', n_repeat, 2, 10, 0.01, None, 200, args)
    # perform predictions after obtaining the labels
    scores = test_prediction(data, True, df_info, icd_label, subid, scale_dict, curve_range, te_sz, 'LR', n_repeat, 2, 10, 0.01, None, 200, args)
    print('Length of time:', args)
    print(scores)
    return ;

def main(num_job=4):
    #beta = [1e-2, 1e-1, 1, 10]
    #gamma = [1e-3, 1e-2, 1e-1, 1, 10]
    #combinations = list(itertools.product(beta, gamma))
    combinations = [12, 24, 48, 120]
    with Pool(num_job) as pool:
        results = pool.map(wrap_fcn, combinations)

if __name__ == '__main__':
    main()
