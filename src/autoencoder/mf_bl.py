# -*- coding: utf-8 -*-
'''
Created on 2018年12月13日

@author: zwp
'''

import numpy as np;
import time;
import math;
import os;
from tools import SysCheck;
from tools.fwrite import fwrite_append;
import pickle as pk;
from autoencoder.ModelClass import MF_bl_ana


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';

NoneValue= 0;
# 初始化参数中 正态分布标准差
rou = 0.1;
# 在矩阵分解中 正则化 参数
lamda = 0.1;

# 隐属性数
f = 100;

#训练次数
epoch = 56
# 学习速率
learn_rate = 0.012;
# 
batch_size=1;

spas=[5]

us_shape=(339,5825);
case = [1,2,3,4,5];
loadvalues=False;
continue_train=True;

def data_splite(d):
    us = d[:,0:2].astype(np.int32);
    r = d[:,2].astype(np.float32);
    return us,r;

def mf_bl_run(spa,case):
    train_data = base_path+'/Dataset/ws/train_n/sparseness%.1f/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test_n/sparseness%.1f/test%d.txt'%(spa,case);
       
    model_save_path=base_path+'/Dataset/mf_baseline_values/model_spa%.1f_case%d.dp'%(spa,case);
    
    print('开始实验，稀疏度=%.1f,case=%d'%(spa,case));
    print ('加载训练数据开始');
    now = time.time();
    trdata = np.loadtxt(train_data, dtype=float);
    n = np.alen(trdata);
    print ('加载训练数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));
    
    print ('加载测试数据开始');
    tnow = time.time();
    ttrdata = np.loadtxt(test_data, dtype=float);
    n = np.alen(trdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),n));    
    
    print ('清除数据开始');
    tnow = time.time();
    idx = np.where(trdata[:,2]>0);
    train_data=trdata[idx];
    idx = np.where(ttrdata[:,2]>0);
    test_data=ttrdata[idx];
    r_mean = np.mean(train_data[:,2]);
    print ('清除数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    print ('预处理数据开始');
    tnow = time.time();
    train_data = data_splite(train_data);
    test_data = data_splite(test_data);
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    
    print ('训练模型开始');
    tnow = time.time();
    
    if loadvalues:
        model = pk.load(open(model_save_path, "rb"));
    else:
        model = MF_bl_ana(us_shape,f,r_mean);
    
    if continue_train:
        model.train_mat(train_data[0], train_data[1], val=test_data,
                repeat=epoch, learn_rate=learn_rate, lamda=lamda)
        pk.dump(model,open(model_save_path, "wb"));
    
    py = model.predictAll(test_data[0][:,0], test_data[0][:,1])
    mae = np.mean(np.abs(py-test_data[1]));
    print(mae);
    return mae;
    
    
if __name__ == '__main__':
    
    for spa in spas:
        res=0;
        for ca in case:
            res+=mf_bl_run(spa,ca);
        out_s = 'mf_bl out spa=%.1f mae=%.6f'%(spa,res/len(case))
        print(out_s);
        fwrite_append('./mf_bl_res.txt',out_s);
    pass