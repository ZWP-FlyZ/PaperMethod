# -*- coding: utf-8 -*-
'''
Created on 2018年4月16日

@author: zwp
'''

'''
    分割 339*5825 数据集，
    从原数据集中采集spa稀疏度的数据作为训练集，
    剩下的数据中取spa稀疏度的数据作为测试集
    将无效值从-1变为0,
'''

import time;
import numpy as np;
import random;
import os;
from sklearn.model_selection import train_test_split;
from tools import SysCheck;

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
    
origin_data_path = base_path+'/Dataset/ws/tpmatrix.txt';
train_output_path = base_path+'/Dataset/ws/train_tp'
test_output_path = base_path+'/Dataset/ws/test_tp'

spa_list=[5,10,15,20];
case_cout=10;
replace_param=[-1,-1];



def load_origin_data(ori_path,repalce_param=None):
    '''
    加载一个原始数据集R,[339,5825],
    repalce_param:将R中所有[0]的值替换为[1];
    '''
    R = np.loadtxt(ori_path,float);
    if repalce_param != None:
        ind = np.where(R==repalce_param[0]);
        R[ind] = repalce_param[1];
    return R;

def mat_to_list(R):
    '''
    将一个矩阵形式数据按照各个维度展开，
    各个轴index从0开始
    返回属性列表和标签列表
    '''
    us_shape = R.shape;
    feature=[];
    lable=[];
    for i in range(us_shape[0]):
        for j in range(us_shape[1]):
            feature.append([i,j]);
            lable.append(R[i,j]);
    return np.array(feature),np.array(lable)

def run():
    
    print('开始分割!分割序列=',spa_list);
    print ('加载数据开始');
    now = time.time();
    R = load_origin_data(origin_data_path,replace_param);
    print('原始数据：\n',R);
    print ('加载数据完成，耗时 %.2f秒\n'%((time.time() - now)));
    
    print ('转换数据开始');
    tnow = time.time();
    feature,lable=mat_to_list(R);
    n = len(feature);
    print ('转换数据开始，耗时 %.2f秒,总数据%d\n'%((time.time() - tnow),n));
    
    
    for spa in spa_list:
        d_size = int(spa / 100.0 * n);
        test_path = test_output_path+'/sparseness%.1f'%(spa);
        if not os.path.isdir(test_path):
            os.makedirs(test_path);
        train_path = train_output_path+'/sparseness%.1f'%(spa);
        if not os.path.isdir(train_path):
            os.makedirs(train_path);        
        for case in range(1,case_cout+1):
            print ('-->开始生成稀疏度%.1f%%数据,数据量%d,case=%d'%(spa,d_size,case));
            tnow = time.time();
            td_size = int(d_size/10);
            # td_size = int(d_size);
            test_x,left_x,test_y,left_y = train_test_split(feature,lable,train_size=td_size);
            test_y = test_y.reshape([td_size,1]);
            new_test=np.hstack((test_x,test_y));
            del test_y;
            del test_x;
            test_file = test_path+'/test%d.txt'%(case);
            np.savetxt(test_file,new_test,'%d %d %.2f');
            del new_test;
            
            train_x,_,train_y,_ = train_test_split(left_x,left_y,train_size=d_size);
            train_y = train_y.reshape([d_size,1]);
            new_train=np.hstack((train_x,train_y));
            del train_x;
            del train_y;
            train_file = train_path+'/training%d.txt'%(case);
            np.savetxt(train_file,new_train,'%d %d %.2f');
            del new_train;            
    pass;

if __name__ == '__main__':
    run();
    pass