# -*- coding: utf-8 -*-
'''
Created on 2018年5月7日

@author: zwp12
'''

import numpy as np;
import time;
import math;
import os;
from tools import SysCheck
import matplotlib.pyplot as plt 

def preprocess(R):
    if R is None:
        return R;
    ind = np.where(R>0);
    newR = np.zeros_like(R);
    newR[ind]=1;
    return  newR; 


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/Dataset/ws/rtmatrix.txt';

us_shape=(339,5825);

mean = 0.908570086101;
ek = 1.9325920405;
# 训练例子
spas=[10]
case = 1;
NoneValue = 0.0;


fid = 1;
def setFigure(X,Y,fid):
    plt.figure(fid);
    plt.grid();
    plt.plot(X,Y);
def showFig():
    plt.show();


def encoder_run(spa):
    train_data = base_path+'/Dataset/ws/train_n/sparseness%d/training%d.txt'%(spa,case);
    
    test_data = base_path+'/Dataset/ws/test_n/sparseness%d/test%d.txt'%(spa,case);
    W_path = base_path+'/Dataset/ws/BP_CF_W_spa%d_t%d.txt'%(spa,case);
    loc_path = base_path+'/Dataset/ws';   
    values_path=base_path+'/Dataset/ae_values_space/spa%d'%(spa);
    
    print('开始实验，稀疏度=%d,case=%d'%(spa,case));
    print ('加载训练数据开始');
    now = time.time();
    trdata = np.loadtxt(train_data, dtype=float);

    n = np.alen(trdata);
    print ('加载训练数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));
    
    print ('转换数据到矩阵开始');
    tnow = time.time();
    u = trdata[:,0];
    s = trdata[:,1];
    u = np.array(u,int);
    s = np.array(s,int);
    R = np.full(us_shape, NoneValue, float);
    R[u,s]=trdata[:,2];
    
    R = np.loadtxt(origin_data, dtype=float);
    

    print ('转换数据到矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    print ('预处理数据开始');
    tnow = time.time();
    ## 清除 复数数据
    idx = np.where(R<0);
    R[idx]=0;
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));

    r_list = np.reshape(R,(-1,));
    r_list = r_list[np.where(r_list>0)];
    mean = np.mean(r_list);
    std = np.std(r_list);
    print(mean,std);
    

    print ('统计数值分布开始  \n');
    
    tnow = time.time()
    step_range=1500;
    step = 20.0 / step_range;
    x_list= np.arange(20,step=step);
    ## 统计数值分布
    boxes = np.zeros((step_range,),float);
    for u in range(us_shape[0]):
        for s in range(us_shape[1]):
            rt = R[u,s];
            if rt==0.0:continue;
            bid = int(rt/step);
            boxes[bid]+=1;
#     setFigure(x_list, boxes, spa+1);
    print ('统计数值分布，耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    
    print ('计算用户欧式距离分布图  \n');
    ## 计算用户欧式距离分布图
    path = base_path+'/Dataset/ws/user_distance.txt'
    tnow = time.time()
    step_range=8000;
    max_v = 8;
    step = max_v / step_range;
    x_list= np.arange(max_v,step=step);
    uboxes = np.zeros((step_range,),float);
    UW = np.zeros((us_shape[0],us_shape[0]))
    for u in range(us_shape[0]):
        if u%60 ==0:
            print('step%d ucf_w'%u);
        a = R[u];
        for v in range(u+1,us_shape[0]):
            b = R[v];
            log_and = (a!=0) & (b!=0);
            cot = np.count_nonzero(log_and);
            delta = np.subtract(a,b,out=np.zeros_like(a),where=log_and);
            dis = np.sqrt(np.sum(delta**2)/cot);
#             dis = np.sqrt(np.sum(delta**2))/us_shape[1];
            UW[u,v]=UW[v,u]=dis;
            if dis==0.0:continue;
            bid = int(dis/step);
            uboxes[bid]+=1;
            pass
    np.savetxt(path, UW, '%.6f');
    setFigure(x_list, uboxes, spa+2);
    print ('计算用户欧式距离分布图，耗时 %.2f秒  \n'%((time.time() - tnow)));        
    

    
    
    showFig()
    


if __name__ == '__main__':
    
    for spa in spas:
        encoder_run(20);
    pass