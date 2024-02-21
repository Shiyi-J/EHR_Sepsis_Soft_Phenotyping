import random
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from Classification_RNN import *
from utils import *
from scipy.stats import skew
import matplotlib.pyplot as plt
from textwrap import wrap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def gen_subj_data(data, name, subj_gr, length, time):
    subj_data = []
    for id in subj_gr:
        features = []
        s1, hr_val = format_data(data, id, name, length, time)
        '''
        s2 = s1[:length*0.1]
        s3 = s1[:length*0.25]
        s4 = s1[:length*0.5]
        s5 = s1[length*0.5:]
        s6 = s1[length*0.75:]
        s7 = s1[length*0.9:]
        '''
        # no. of values
        features.append(np.count_nonzero(s1))
        features.append(np.count_nonzero(s1[:math.ceil(length*0.1)]))
        features.append(np.count_nonzero(s1[:math.ceil(length*0.25)]))
        features.append(np.count_nonzero(s1[:math.ceil(length*0.5)]))
        features.append(np.count_nonzero(s1[math.ceil(length*0.5):]))
        features.append(np.count_nonzero(s1[math.ceil(length*0.75):]))
        features.append(np.count_nonzero(s1[math.ceil(length*0.9):]))
        # subsequences
        s1 = interplt(hr_val, length)
        s2 = s1[:math.ceil(length*0.1)]
        s3 = s1[:math.ceil(length*0.25)]
        s4 = s1[:math.ceil(length*0.5)]
        s5 = s1[math.ceil(length*0.5):]
        s6 = s1[math.ceil(length*0.75):]
        s7 = s1[math.ceil(length*0.9):]
        for seq in [s1, s2, s3, s4, s5, s6, s7]:
            features.append(max(seq))
            features.append(min(seq))
            features.append(np.mean(seq))
            features.append(np.std(seq))
            features.append(skew(seq))
        subj_data.append(features)
    return subj_data
