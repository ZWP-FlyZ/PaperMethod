# -*- coding: utf-8 -*-
'''
Created on 2018年12月13日

@author: zwp
'''

import numpy as np;
import time;
import math;
import os;
from tools import SysCheck
from autoencoder.ModelClass import DenoiseAutoEncoder,MF_bl_ana,CF;
import pickle as pk;
import random;
from tools.LoadLocation import loadLocation
from autoencoder.keras_mfbl import batch_size
from tools.fwrite import fwrite_append


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';

us_shape=(339,5825);
# 是否基于用户的自编码器，预测每个用户的所有服务值
isUserAutoEncoder=True;
# 是否基于服务的CF方法
isICF=False;



# 训练例子
spas=[10,20]
case = [1,2,3,4,5];
NoneValue = 0.0;

# autoencoder 参数
hidden_node = 80;
learn_rate=0.4
learn_param = [learn_rate,100,1.0];
## 15%ep=100,[400,200,100]
repeat = 350;
rou=0.1

# 协同过滤参数
k = 11;
sk = 17;
def get_cf_k(spa):
    if   spa==2.5:  return 200;
    elif spa==5.0:  return 150;
    elif spa==10.0: return 100;
    elif spa==15.0: return 50;
    else:           return 50;
def get_cf_sk(spa):
    if   spa==2.5:  return 300;
    elif spa==5.0:  return 200;
    elif spa==10.0: return 150;
    elif spa==15.0: return 150;
    else:           return 150; 

def get_epoch(spa):
    if   spa==2.5:  return 350;
    elif spa==5.0:  return 350;
    elif spa==10.0: return 200;
    elif spa==15.0: return 150;
    else:           return 110;


loc_w= 1.0;


#### 使用mf填补 
use_mf = True;

# 加载AutoEncoder
use_ae=True;
loadvalues= True;
continue_train = False;


use_cf=True;
use_cf_mode = 2; # 1:UCF 2:SCF
cf_loadmode=False;
cf_continue_train=True;



# 随机删除比率
cut_rate = 0;


#预处理填补比例
def out_cmp_rat(spa):
#     return spa/100;
    if spa<5:return 0.025;
    elif spa==5:return 0.03;
    elif spa==10:return 0.15;
    elif spa==15:return 0.20;
    elif spa==20:return 0.25;
'''
1%:(1.05) +spa->0.700 +5 ->0.75
2%:(0.605)
2.5%:(0.590)+3.5->0.578   0.5 +20->0.515 +40->0.517 +60->0.512 +80->0.53 +95->0.545
3%:(0.550)
4%:(0.528)
5%:+5->0.531 +20->0.515 +40->0.517 +60->0.512 +80->0.53 +95->0.545
10%:(0.472)+5->0.472 +10->0.463 +20->0.460 +40->0.462 +80->0.460 +90->0.466
15%:(0.454)+5->0.472 +15->0.447 +30->0.450 +45->0.447 +80->0.460 +90->0.448
20%:()
'''




# 特征权重约束系数
w_d=50;
sw_d=100;


# 地理位置表
loc_tab=None;



def mf_rat_in(R,mf,rat):
    batch_size,feat_size = R.shape;
    sum_arr = np.count_nonzero(R,axis=0);

    most = int(np.median(sum_arr));
    if rat<=0.0:
        delta = 1;
        top = most;
    else:
        top = int(rat*batch_size);
        
    delta = top-sum_arr;
    all_range= np.arange(batch_size,dtype=np.int);
    for feat in range(feat_size):
        if delta[feat]<=0:continue;
        batch_ind = random.sample(all_range.tolist(),int(delta[feat]));
        for bid in batch_ind:                
            R[bid,feat]=mf.predict(bid, feat);    

def mf_rat_in2(R,mf,rat,seed=2121212):
    '''
    非重复填补
    '''
    batch_size,feat_size = R.shape;
    sum_arr = np.count_nonzero(R,axis=0);
    ivkrec = [];
    for i in range(feat_size):
        ivkrec.append(np.nonzero(R[:,i])[0]);

    most = int(np.median(sum_arr));
    if rat<=0.0:
        delta = 1;
        top = most;
    else:
        top = int(rat*batch_size);
        
    delta = top-sum_arr;
    random.seed(seed);
    all_range= np.arange(batch_size,dtype=np.int);
    for feat in range(feat_size):
        if delta[feat]<=0:continue;
        diffset = np.setdiff1d(all_range, ivkrec[feat]);
        batch_ind = random.sample(diffset.tolist(),int(delta[feat]));
        for bid in batch_ind:                
            R[bid,feat]=mf.predict(bid, feat);



def random_empty(R ,cut_rat,NoneValue=0,seed=121212):
    ind = np.argwhere(R!=NoneValue);
    data_size = len(ind);
    random.seed(seed);
    cut_ind = random.sample(range(data_size),int(cut_rat*data_size));
    cut_us = ind[cut_ind];
    for u,s in cut_us:
        R[u,s]=NoneValue;
    pass;

    
def actfunc1(x):
    return 1.0/( 1.0 + np.exp(np.array(-x,np.float64)))
def deactfunc1(x):
    return x*(1.0-x);


def run(spa,case):
    train_data = base_path+'/Dataset/ws/train_n/sparseness%.1f/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test_n/sparseness%.1f/test%d.txt'%(spa,case);
    W_path = base_path+'/Dataset/ws/BP_CF_W_spa%.1f_t%d.txt'%(spa,case);
    SW_path = base_path+'/Dataset/ws/BP_CF_SW_spa%.1f_t%d.txt'%(spa,case);
    loc_path = base_path+'/Dataset/ws';   
    aemodel_path=base_path+'/Dataset/dae_values/model_spa%.1f_case%d_mf%s.dp'%(spa,case,use_mf);
    if cut_rate==0:
        cf_model_path = base_path+'/Dataset/cf_values/model_spa%.1f_case%d_mf%s_cfmod%d.dp'%(spa,case,use_mf,use_cf_mode);
    else:
        cf_model_path = base_path+ \
                '/Dataset/cf_values/model_spa%.1f_case%d_mf%s_cut%.1f_cfmod%d.dp'%(spa,case,use_mf,cut_rate,use_cf_mode);
    
    mf_model_path=base_path+'/Dataset/mf_baseline_values/model_spa%.1f_case%d.dp'%(spa,case);


    if use_cf_mode==1:
        near_lab_k=get_cf_k(spa);
    else:
        near_lab_k=get_cf_sk(spa);


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
    
    R = np.zeros(us_shape,np.float);
    valR = np.zeros_like(R);
    R[train_data[:,0].astype(np.int),
      train_data[:,1].astype(np.int)] =  train_data[:,2];
    valR[test_data[:,0].astype(np.int),
      test_data[:,1].astype(np.int)] =  test_data[:,2];    
    print(np.count_nonzero(valR))
    del train_data,test_data,idx,trdata,ttrdata;
    print ('清除数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    print ('预处理数据开始');
    tnow = time.time();
    oriR = R.copy();
    if use_mf:
        with open(mf_model_path, "rb") as f:
            mf_model = pk.load(f);
        # 填补处理
        cmp_rat = out_cmp_rat(spa);
        print(cmp_rat);
        mf_rat_in2(R,mf_model,rat=cmp_rat);        
    print(np.sum(R-oriR));
    
    # 归一化
    R/=20.0;# 归一化
    oriR/=20.0;
    valR/=20.0;
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    print ('训练模型开始');
    tnow = time.time();
    ae_val_res=[];
    if loadvalues :
        with open(aemodel_path,"rb") as f:
            dae_model = pk.load(f);
    else :
        dae_model = DenoiseAutoEncoder(us_shape[1]
                            ,hidden_node,
                            actfunc1,deactfunc1,
                             actfunc1,deactfunc1);
    if continue_train:
        ae_val_res = dae_model.train(R, oriR,valR,learn_param,get_epoch(spa));
        print(min(ae_val_res),ae_val_res);
        with open(aemodel_path,"wb") as f:
            pk.dump(dae_model,f);
    
    PR = dae_model.calFill(R);
#     print(R);
#     print();
#     print(PR);
#     print();
############# PR 还原处理   ###############
    PR = PR * 20.0;
    R = R * 20;
    oriR=oriR*20;
    valR = valR*20;
    PR = np.where(R!=NoneValue,R,PR); 
############# PR 还原处理   ###############        
    print ('训练模型结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    if not use_cf:
        return ae_val_res[-1]*20,0;
    
    
    print ('随机删除开始');
    tnow = time.time();
    if use_cf_mode==3:
        R_forUcf = PR;
        R_forScf = PR.copy();
        random_empty(R_forUcf, 0.6);
        random_empty(R_forScf, 0.4);
        PR = (R_forUcf,R_forScf);
    else:
        random_empty(PR, cut_rate);
    print ('随机删除开始，耗时 %.2f秒  \n'%((time.time() - tnow)));    



    print ('训练CF开始');
    if not use_ae:
        PR = R;
    tnow = time.time();
    if cf_loadmode:
        cf_mode = pk.load(open(cf_model_path,"rb"));
    else: 
        cf_mode = CF(us_shape,use_cf_mode);
        
    if cf_continue_train:
        cf_mode.train(PR, oriR,near_lab_k);
        pk.dump(cf_mode,open(cf_model_path,"wb"));
    elif use_cf_mode==1:
        cf_mode.ucf_S(near_lab_k);
    elif use_cf_mode==2:
        cf_mode.scf_S(near_lab_k);

    mae,nmae = cf_mode.evel(valR, oriR);
    print(mae,nmae);
    print ('训练CF结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
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
        out_s = 'mf_ae_cf out mf-ae-cf=(%s,%s,%s) spa=%.1f mae=%.6f nmae=%.6f time=%s'%(use_mf,use_ae,use_cf,sp,s/cot,s2/cot,time.asctime());
        fwrite_append('./mf_ae_cf_res.txt',out_s);
        print(out_s);
    pass