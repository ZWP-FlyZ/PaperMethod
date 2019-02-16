# -*- coding: utf-8 -*-
'''
Created on 2019年1月11日

@author: zwp
'''



'''

基于上下文的用户 模糊聚类
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
fcm_w_out = base_path+'/Dataset/ws/localinfo/user_fcm_w.txt';


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
    data[:,2]=data[:,2]/100.0;
    data[:,3]=data[:,3]/100.0;
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


using_cos_dis=False;

class Fcm():
    '''
    模糊聚类方法
    '''
    clu = 1;# 聚类数
    m = 2;# 平缓系数
    U = None;# 隶属度矩阵
    center=None;# 簇中心
    tmp_dis_a=None;
    def __init__(self,c,m):
        self.clu = c;
        self.m = m;
    
    def _dis(self,a,b):
        '''
        
        '''
        if using_cos_dis:
        # 余弦距离
            dis_b = np.sqrt(np.sum(b**2));
            if self.tmp_dis_a is None:
                dis_a = np.sqrt(np.sum(a**2,axis=1));
                self.tmp_dis_a = dis_a;
            tmp = np.sum(a*b,axis=1)/(self.tmp_dis_a * dis_b);
            return 1-tmp;
        # 欧式距离
        return np.sqrt(np.sum((a-b)**2,axis=1));
    
    def _update_U(self,data,uidx,cent,di=1):
        m = self.m;
        clu = self.clu;
        tmp_u = np.zeros(self.clu);
        clu_rec=[0]*clu;
        clu_result=[[] for _ in range(clu)];
        new_cent = np.zeros_like(cent);
        new_cent_dw = np.zeros(clu);
        for idx in uidx:
            xj = data[idx];
            dis_lg = haversine(cent[:,0],cent[:,1],xj[0],xj[1],r=di);
            dis_lg = np.reshape(dis_lg,[-1,1])**2;
            dis_oth = self._dis(cent[:,2:],xj[2:]).reshape(-1,1)**2;            
            
            v = np.concatenate((dis_lg,dis_oth),axis=1);
            v = np.sum(v,axis=1);

            
            v = v**(1.0/(m-1.0));
            for c in range(clu):
                tt = np.divide(v[c],v,out=np.ones_like(v),where=v!=0);
                tt = np.sum(tt)**(-1);
                tmp_u[c] = tt;
                tt = tt**m;
                new_cent[c] += xj*tt;
                new_cent_dw[c] += tt;
                
            max_idx = np.argmax(tmp_u);
            clu_result[max_idx].append(idx);
            clu_rec[max_idx] = clu_rec[max_idx]+1;
            self.U[idx]= tmp_u;
        
        new_cent = new_cent/ new_cent_dw.reshape((-1,1));
        return new_cent,clu_rec,clu_result;
        pass;
        
    def train(self,data,max_loop=100,max_e = 0.00001,di=1):
        '''
        '''
        ds = len(data);
        epo = 0;# 迭代次数
        updata_idx = np.arange(ds,dtype=np.int);
        clu = self.clu;
        # 初始化U
        self.U = np.zeros((ds,clu));

        # 归一化 IP
        data[:,2]=data[:,2]/150.0;
        data[:,3]=data[:,3]/150.0;


        # 初始化中心点
        eidx = np.random.choice(updata_idx,size=clu,
                                    replace=False);
        self.center = data[eidx];
  
        while epo<max_loop:
            np.random.shuffle(updata_idx); # 随机训练顺序
            new_cent,rec,clu_res = self._update_U(data, updata_idx, self.center,di);
            err = np.sum(np.abs(new_cent-self.center));
            epo+=1;
            print('epoch=%d \terr=%.6f \tclu_ele_rec=%s'%(epo,err,str(rec)));
            print();
            self.center=new_cent;
            self.tmp_dis_a=None;
            if err<=max_e:break;
        print(self.U);
        
        return new_cent,clu_res;


def run():
    
    user_loc = localload.load_userinfo(user_info_path)
    user_loc_m = localload.load_locmore(user_info_more_path);
    R = np.loadtxt(origin_path,np.float);
    if os.path.isfile(fcm_w_out):
        os.remove(fcm_w_out);

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
    k=4;
    for uid in range(339):
        un = user_loc[uid][1];
        names.append(un);
        area.append(user_loc_m[un][0])
        lc = [];
        lc.extend(user_loc[uid][2]);
        lc.extend(user_loc[uid][3][:2]);#ip前两域
        lc.append(user_mean[uid]);# 均值
        
        
        data.append(lc);
    data=np.array(data);

    # 1.2-1.3
    fcm = Fcm(k,1.7);
    cent,res = fcm.train(data, max_loop=100, max_e=0.00001,di=k)

#     cent,res = simple_km2(data,k,k);
    
    print(cent);
    print(res);
    np.savetxt(fcm_w_out, fcm.U, '%.6f');
    
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