# -*- coding: utf-8 -*-
'''
Created on 2018年9月14日

@author: zwp12
'''


import numpy as np;
import time;
import math;
import os;
from tools import SysCheck;

from content_ncf.ncf_param import NcfTraParm,NcfCreParam;
from content_ncf.ncf_models import ncf_pp_local;

from content_ncf import localtools;

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';


'''
spa20
uclass 1:
    class 6 :0.35-0.37
    class12 :效果变差
    class4:0.355-0.37
    class3: 效果变好 0 .34
    class2:不好
    class1:常规

uclass 3:
    class 6 :比3好一些
    class12 :
    class4
    class3:变差
    class2:
    class1:


'''


spas = [20];


def mf_base_run(spa,case):
    train_path = base_path+'/Dataset/ws/train_n/sparseness%.1f/training%d.txt'%(spa,case);
    test_path = base_path+'/Dataset/ws/test_n/sparseness%.1f/test%d.txt'%(spa,case);
    cache_path = 'value_cache/spa%d_case%d.ckpt'%(spa,case);
    result_file= 'result/ws_spa%.1f_case%d.txt'%(spa,case);
    dbug_paht = 'E:/work/Dataset/wst64/rtdata1.txt';
    
    loc_classes = base_path+'/Dataset/ws/localinfo/ws_content_classif_out.txt';
    
    print('开始实验，稀疏度=%.1f,case=%d'%(spa,case));
    print ('加载训练数据开始');
    now = time.time();
    trdata = np.loadtxt(train_path, dtype=float);
    ser_class = localtools.load_classif(loc_classes);
    classiy_size = len(ser_class);
    n = np.alen(trdata);
    print ('加载训练数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));
    
    
    print ('加载测试数据开始');
    tnow = time.time();
    ttrdata = np.loadtxt(test_path, dtype=float);
    tn = np.alen(ttrdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),tn));
    
    print ('分类数据集开始');
    tnow = time.time();
    train_sets = localtools.data_split_class(ser_class, trdata);
    test_sets = localtools.data_split_class(ser_class, ttrdata);
    print ('分类数据集结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    
    
    cp = NcfCreParam();
    tp = NcfTraParm();
    cp.us_shape=(339,5825);
    cp.hid_feat=32;
    cp.hid_units=[64,32,16];
    cp.drop_p=0.00001
    cp.reg_p=0.0001
    
    # 处理用户访问服务记录 
    R = np.zeros(cp.us_shape);
    u = trdata[:,0].astype(np.int32);
    s = trdata[:,1].astype(np.int32);
    R[u,s]=1.0;
    us_invked = [];
    for cla in ser_class:
        hot = np.zeros([cp.us_shape[1]],np.float32);
        hot[cla]=1.0;
        usi = R*hot;
        nonzeroes = np.sqrt(np.count_nonzero(usi, axis=1));
        noz = np.divide(1.0,nonzeroes,
                        out=np.zeros_like(nonzeroes),where=nonzeroes!=0);
        noz = np.reshape(noz,[-1,1]);
        us_invked.append((usi*noz).astype(np.float32));
    
     
        
    tp.train_data=train_sets;
    tp.test_data=test_sets;
    tp.epoch=40;
    tp.batch_size=5;
    tp.learn_rate=0.007;
    tp.lr_decy_rate=1.0
    tp.lr_decy_step=int(n/tp.batch_size);
    tp.cache_rec_path=cache_path;
    tp.result_file_path= result_file;
    tp.load_cache_rec=False;
    tp.classif_size = len(train_sets);
    tp.us_invked= us_invked;
    
    
    print ('训练模型开始');
    tnow = time.time();
    model = ncf_pp_local(cp);
    
    model.train(tp);
                     
    print ('训练模型结束，耗时 %.2f秒  \n'%((time.time() - tnow)));  

   

    print('实验结束，总耗时 %.2f秒,稀疏度=%.1f\n'%((time.time()-now),spa));


if __name__ == '__main__':
    for spa in spas:
        for ca in range(1,2):
            case = ca;
            mf_base_run(spa,case);



if __name__ == '__main__':
    pass