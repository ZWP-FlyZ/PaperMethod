# -*- coding: utf-8 -*-
'''
Created on 2018年11月19日

@author: zwp12
'''

'''

基于上下文的用户聚类
'''


import numpy as np;
import time;
import random;
import os;
from tools import SysCheck;
from tools import localload;
from tools import utils;
from tools import fwrite;


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work';
origin_path = base_path+'/Dataset/ws/rtmatrix.txt';
user_info_path=base_path+'/Dataset/ws/localinfo/user_info.txt';
user_info_more_path=base_path+'/Dataset/ws/localinfo/user_info_more.txt';
loc_class_out = base_path+'/Dataset/ws/localinfo/ws_content_classif_out_by_user.txt';

def simple_km(data,k,di=1.0):
    datasize = len(data);
    di = float(di);
    if k<1 or  datasize<k:
        raise ValueError('data,k err');
    rep=0;
    edg=180.0/di;
    edg2=  edg*2;
    data[:,0]=data[:,0]/di;
    data[:,1]=data[:,1]/(di*2.0);
    cents=data[random.sample(range(0,datasize),k)];
    last_c = cents;
    
    while True:
        res = [[] for _ in range(k)];
        for i in range(datasize):
            dis = np.abs(cents-data[i]);
            dis[:,1]=np.where(dis[:,1]>edg,edg2-dis[:,1],dis[:,1]);
            dis = np.sum(dis**2,axis=1);
            tagk= np.argmin(dis);
            res[tagk].append(i);
        last_c = np.copy(cents);
        for i in range(k):
            cents[i]=np.mean(data[res[i]],axis=0);    
        bout = np.sum(cents-last_c);
        if bout==0:break;
        rep+=1;
        if rep%1 == 0:
            print('rep=%d,delta=%f'%(rep,bout));
        

    return cents,res;
    pass;




'''
haversine 公式
'''



def haversine( lat1, lon1, lat2,lon2,r=6371): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    """
    # 将十进制度数转化为弧度
    lon1,lat1,lon2, lat2 = map(np.radians, [lon1,lat1,lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = r # 地球平均半径，单位为公里
    return c * r # 输出为公里



def simple_km2(data,k,di=1.0):
    datasize = len(data);
    di = float(di);
    if k<1 or  datasize<k:
        raise ValueError('data,k err');
    rep=0;
#     edg=180.0/di;
#     edg2=  edg*2;

#     IP 域归一化
    data[:,2]=data[:,2]/150.0;
    data[:,3]=data[:,3]/150.0;
    cents=data[random.sample(range(0,datasize),k)];
    last_c = cents;
    
    while True:
        res = [[] for _ in range(k)];
        for i in range(datasize):
            
            
            # 计算距离
            dis_lg = haversine(cents[:,0],cents[:,1],data[i,0],data[i,1],r=di);
            dis_lg = np.reshape(dis_lg,[-1,1])**2;
            dis_oth = np.abs(cents[:,2:]-data[i,2:])**2;
            
            dis = np.concatenate((dis_lg,dis_oth),axis=1);
            dis = np.sum(dis,axis=1);
            
            tagk= np.argmin(dis);
            res[tagk].append(i);
        last_c = np.copy(cents);
        for i in range(k):
            cents[i]=np.mean(data[res[i]],axis=0);    
        bout = np.sum(cents-last_c);
        if bout==0:break;
        rep+=1;
        if rep%1 == 0:
            print('rep=%d,delta=%f'%(rep,bout));
        

    return cents,res;
    pass;





def classf(carr,tagdir):
    res = [];
    for idx in tagdir:
        if tagdir[idx][1] in carr:
            res.append(idx);
    fwrite.fwrite_append(loc_class_out, utils.arr2str(res));


def write2file(res):
    for li in res:
        fwrite.fwrite_append(loc_class_out, utils.arr2str(li));

def run():
    
    user_loc = localload.load_userinfo(user_info_path)
    user_loc_m = localload.load_locmore(user_info_more_path);
    R = np.loadtxt(origin_path,np.float);
    if os.path.isfile(loc_class_out):
        os.remove(loc_class_out);

    idx = np.where(R<0);
    R[idx]=0;
    
    user_sum = np.sum(R,axis=1);
    user_cot = np.count_nonzero(R, axis=1);
    user_mean = np.divide(user_sum,user_cot,
        out=np.zeros_like(user_sum),where=user_cot!=0);
    all_mean = np.sum(user_sum)/np.sum(user_cot);
    user_mean[np.where(user_cot==0)] = all_mean;
    
    data=[];
    names=[];
    area=[];
    k=3;
    for uid in range(339):
        un = user_loc[uid][1];
        names.append(un);
        area.append(user_loc_m[un][0])
        lc = [];
        lc.extend(user_loc[uid][2]);
        lc.extend(user_loc[uid][3][:2]);#ip前两域
        lc.append(user_mean[uid]);
        
        
        data.append(lc);
    data=np.array(data);

    cent,res = simple_km2(data,k,k);
    
    print(cent);
    print(res);
    write2file(res);
    
    for i in range(k):
        tmp=[];
        tmp2=[];
        for id in res[i]:
            if names[id] not in tmp2:
                tmp2.append(names[id]);
                tmp.append(area[id]);
        print(tmp)
        print(tmp2);
        print();
        

    pass;

if __name__ == '__main__':
    run();
    pass
