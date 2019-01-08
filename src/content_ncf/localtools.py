# -*- coding: utf-8 -*-
'''
Created on 2018年9月5日

@author: zwp12
'''

from tools import utils;
import numpy as np;

def load_classif(class_path):
    ret = [];
    with open(class_path) as f:
        for line in f:
            ret.append(utils.str2arr(line.strip()));
    return  ret;

def data_split_class(ces,data):
    k = len(ces);
    ret = [[] for _ in range(k)];
    for d in data:
        ci = 0;
        while ci<k:
            if int(d[1]) in ces[ci]:
                break;
            ci+=1;
        if ci<k:
            ret[ci].append(d);
    for i in range(k):
        ret[i] = np.array(ret[i]);
        
    return ret;

def data_split_class_byuser(ces,data):
    k = len(ces);
    ret = [[] for _ in range(k)];
    for d in data:
        ci = 0;
        while ci<k:
            if int(d[0]) in ces[ci]:
                break;
            ci+=1;
        if ci<k:
            ret[ci].append(d);
    for i in range(k):
        ret[i] = np.array(ret[i]);
        
    return ret;




