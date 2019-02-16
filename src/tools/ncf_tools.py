# -*- coding: utf-8 -*-
'''
Created on 2018年8月7日

@author: zwp12
'''

import numpy as np;

def del_non_data(rec_table):
    
    ind = np.where(rec_table[:,2]>0)[0];
    return rec_table[ind,:];

def del_non_data3D(rec_table):
    
    ind = np.where(rec_table[:,3]>0)[0];
    return rec_table[ind,:];

def reoge_data(rec_table):
    rec_table = np.array(rec_table);
    rec_table = del_non_data(rec_table);
    return rec_table[:,0:2].astype(int),    \
        rec_table[:,2].reshape((-1,1)).astype(np.float32)

def reoge_data_for_keras(rec_table):
    rec_table = np.array(rec_table);
    rec_table = del_non_data(rec_table);
    return [rec_table[:,0].astype(int),rec_table[:,1].astype(int)],    \
        rec_table[:,2:3].astype(np.float32)


def reoge_data_for_context_(rec_table,fcm_ws):
    user_w,service_w = fcm_ws;
    rec_table = np.array(rec_table);
    rec_table = del_non_data(rec_table);
    u = rec_table[:,0].astype(int);
    s = rec_table[:,1].astype(int);
    u_cw =  user_w[u];
    s_cw = service_w[s];
    return [u_cw,s_cw],    \
        rec_table[:,2:3].astype(np.float32)
        
def reoge_data_for_context(rec_table,fcm_ws):
    user_w,service_w = fcm_ws;
    rec_table = np.array(rec_table);
    rec_table = del_non_data(rec_table);
    u = rec_table[:,0].astype(int);
    s = rec_table[:,1].astype(int);
    u_cw =  user_w[u];
    s_cw = service_w[s];
    return [u,s,u_cw,s_cw],    \
        rec_table[:,2:3].astype(np.float32)

def reoge_data3D(rec_table):
    rec_table = np.array(rec_table);
    rec_table = del_non_data3D(rec_table);
    return rec_table[:,0:3].astype(int),    \
        rec_table[:,3].reshape((-1,1)).astype(np.float32)
    

if __name__ == '__main__':
    
    a = np.array([[1,2,3],[2,3,4],[0,3,-1]])
    print(reoge_data(a));
    
    
    
    pass