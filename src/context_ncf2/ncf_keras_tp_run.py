# -*- coding: utf-8 -*-
'''
Created on 2019年2月16日

@author: zwp12
'''


import numpy as np;
import time;
import math;
import os;
from tools import SysCheck;

from context_ncf2.ncf_param import NcfTraParm,NcfCreParam;
from context_ncf2.ncf_models import *;
from tools.fwrite import fwrite_append


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';


spas = [5];
case = [1,2,3,4,5];

def run(spa,case):
    train_path = base_path+'/Dataset/ws/train_tp/sparseness%.1f/training%d.txt'%(spa,case);
    test_path = base_path+'/Dataset/ws/test_tp/sparseness%.1f/test%d.txt'%(spa,case);
    cache_path = base_path+'/Dataset/context_ncf_tp_values/spa%.1f_case%d.h5'%(spa,case);
    result_file= './result/tp_ws_spa%.1f_case%d.txt'%(spa,case);
    dbug_paht = 'E:/work/Dataset/wst64/rtdata1.txt';
    
    
    user_fcm_w_path = base_path+'/Dataset/ws/localinfo/user_fcm_w_tp.txt';
    service_fcm_w_path = base_path+'/Dataset/ws/localinfo/service_fcm_w_tp.txt';
    
    print('开始实验，稀疏度=%.1f,case=%d'%(spa,case));
    print ('加载训练数据开始');
    now = time.time();
    trdata = np.loadtxt(train_path, dtype=float);
    user_fcm_w = np.loadtxt(user_fcm_w_path, dtype=float);
    service_fcm_w = np.loadtxt(service_fcm_w_path, dtype=float);
    n = np.alen(trdata);
    print ('加载训练数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));
    
    
    print ('加载测试数据开始');
    tnow = time.time();
    ttrdata = np.loadtxt(test_path, dtype=float);
    tn = np.alen(ttrdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),tn));
    
#     print ('分类数据集开始');
#     tnow = time.time();
#     train_sets = localtools.data_split_class(ser_class, trdata);
#     test_sets = localtools.data_split_class(ser_class, ttrdata);
#     print ('分类数据集结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    
    
    cp = NcfCreParam();
    tp = NcfTraParm();
    cp.us_shape=(339,5825);
    cp.clu_num=(user_fcm_w.shape[1],service_fcm_w.shape[1]);
    cp.hid_feat=16;
    cp.hid_units=[32,16];
    cp.drop_p=0
    cp.reg_p=0
    
    # 处理用户访问服务记录 
    R = np.zeros(cp.us_shape);
    u = trdata[:,0].astype(np.int32);
    s = trdata[:,1].astype(np.int32);
    R[u,s]=trdata[:,2];
    umean = None;
    smean = None;
    R[np.where(R>0)]=1.0;
#     print(umean);
#     print(smean);
#     us_invked = [];
#     for cla in ser_class:
#         hot = np.zeros([cp.us_shape[1]],np.float32);
#         hot[cla]=1.0;
#         usi = R*hot;
#         nonzeroes = np.sqrt(np.count_nonzero(usi, axis=1));
#         noz = np.divide(1.0,nonzeroes,
#                         out=np.zeros_like(nonzeroes),where=nonzeroes!=0);
#         noz = np.reshape(noz,[-1,1]);
#         us_invked.append((usi*noz).astype(np.float32));
    
     
        
    tp.train_data=trdata;
    tp.test_data=ttrdata;
    tp.epoch=30;
    tp.batchsize=5;
    '''
    Adagrad lr 0.03 zui hao
    RMSprop lr 0.005
    
    '''
    
    tp.learn_rate=0.02;
    tp.lr_decy_rate=1.0
    tp.lr_decy_step=int(n/tp.batch_size);
    tp.cache_rec_path=cache_path;
    tp.result_file_path= result_file;
    tp.load_cache_rec=False;
    tp.classif_size = 0;
    tp.us_invked= R;
    tp.umean=umean;
    tp.smean=smean;
    tp.fcm_ws=(user_fcm_w,service_fcm_w);
    
    
    print ('训练模型开始');
    tnow = time.time();
    model = context_ncf(cp);
    
    mae,nmae = model.train(tp);
                     
    print ('训练模型结束，耗时 %.2f秒  \n'%((time.time() - tnow)));  

    print('实验结束，总耗时 %.2f秒,稀疏度=%.1f\n'%((time.time()-now),spa));
    
    return mae,nmae;

if __name__ == '__main__':
    for sp in spas:
        s = 0;s2=0;cot=0;
        for ca in case:
            for i in range(1):
                mae,nmae = run(sp,ca);
                s+=mae;
                s2+=nmae;
                cot+=1;
        out_s = '(NCF-2)spa=%.1f mae=%.6f nmae=%.6f time=%s'%(sp,s/cot,s2/cot,time.asctime());
        print(out_s);
        fwrite_append('./simple_ncf.txt',out_s);