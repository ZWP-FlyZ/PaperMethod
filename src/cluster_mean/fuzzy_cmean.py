# -*- coding: utf-8 -*-
'''
Created on 2019年1月10日

@author: zwp
'''

import numpy as np;
import random;
import matplotlib.pyplot as plt;

using_cos_dis=False;

def distance(a,b,tmp_dis_a):
    
    if using_cos_dis:
        # 余弦距离
        dis_b = np.sqrt(np.sum(b**2));
        if tmp_dis_a is None:
            dis_a = np.sqrt(np.sum(a**2,axis=1));
            tmp_dis_a = dis_a;
        tmp = np.sum(a*b,axis=1)/(tmp_dis_a * dis_b);
        return 1-tmp,tmp_dis_a;
        
    # 欧式距离
    return np.sqrt(np.sum((a-b)**2,axis=1)),None;    

def simple_km(data,k):
    datasize = len(data);
    if k<1 or  datasize<k:
        raise ValueError('data,k err');
    cents=data[random.sample(range(0,datasize),k)];
    last_c = cents;
    while True:
        res = [[] for _ in range(k)];
        tmp_dis_a=None;
        for i in range(datasize):
            dis,tmp_dis_a= distance(cents, data[i], tmp_dis_a);
            tagk= np.argmin(dis);
            res[tagk].append(i);
        last_c = np.copy(cents);
        for i in range(k):
            cents[i]=np.mean(data[res[i]],axis=0);    
        bout = np.sum(cents-last_c);
        if bout==0:break;

    return cents,res;


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
    
    def _update_U(self,data,uidx,cent):
        m = self.m;
        clu = self.clu;
        tmp_u = np.zeros(self.clu);
        clu_rec=[0]*clu;
        clu_result=[-1]*len(data);
        new_cent = np.zeros_like(cent);
        new_cent_dw = np.zeros(clu);
        for idx in uidx:
            xj = data[idx];
            v = self._dis(cent, xj);
            
            v = v**(1.0/(m-1.0));
            for c in range(clu):
                tt = np.divide(v[c],v,out=np.ones_like(v),where=v!=0);
                tt = np.sum(tt)**(-1);
                tmp_u[c] = tt;
                tt = tt**m;
                new_cent[c] += xj*tt;
                new_cent_dw[c] += tt;
                
            max_idx = np.argmax(tmp_u);
            clu_result[idx]=max_idx;
            clu_rec[max_idx] = clu_rec[max_idx]+1;
#             U[idx]= tmp_u;
        
        new_cent = new_cent/ new_cent_dw.reshape((-1,1));
        return new_cent,clu_rec,np.array(clu_result);
        pass;
        
    def train(self,data,max_loop=100,max_e = 0.00001):
        '''
        '''
        ds = len(data);
        epo = 0;# 迭代次数
        updata_idx = np.arange(ds,dtype=np.int);
        clu = self.clu;
        # 初始化U
#         self.U = np.zeros((ds,clu));
        # 初始化中心点
        eidx = np.random.choice(updata_idx,size=clu,
                                    replace=False);
        self.center = data[eidx];
  
        while epo<max_loop:
            np.random.shuffle(updata_idx); # 随机训练顺序
            new_cent,rec,clu_res = self._update_U(data, updata_idx, self.center);
            err = np.sum(np.abs(new_cent-self.center));
            epo+=1;
            print('epoch=%d \terr=%.6f \tclu_ele_rec=%s'%(epo,err,str(rec)));
            print(clu_res);
            print();
            self.center=new_cent;
            self.tmp_dis_a=None;
            if err<=max_e:break;
        print(self.U);
        
        return np.array(clu_res);
        
def get_dataset(cents=[],clz_ds=10,sig=0.1,noise=False):
    
    ce = cents[0];
    tx = np.random.normal(ce[0],sig,size=[clz_ds,1]);
    ty = np.random.normal(ce[1],sig,size=[clz_ds,1]);
    X = np.concatenate([tx,ty],axis=1);
    Y = np.full(clz_ds,0);
    
    for cei in range(1,len(cents)):
        ce = cents[cei];
        tx = np.random.normal(ce[0],sig,size=[clz_ds,1]);
        ty = np.random.normal(ce[1],sig,size=[clz_ds,1]);
        X = np.concatenate([X,np.concatenate([tx,ty],axis=1)],axis=0);
        Y = np.append(Y, np.full(clz_ds,cei));
        pass;        
    
    return X,Y   
    
def show_plt(data,clz_ds,py,py2):
    
    py = np.array(py);
    fig,(ax1,ax2,ax3) = plt.subplots(1, 3);
    
    clz = len(data)//clz_ds;
    start=0;
    c=0;
    cm=['r','g','b','y'];
    mark=['+','*','>','x'];
    ax1.set_title('ORIGIN');
    ax2.set_title('FCM');
    ax3.set_title('HCM');
    while start<len(data):
        
        end = start+clz_ds;
        clz_d = data[start:end];
        ax1.scatter(clz_d[:,0],clz_d[:,1],c=cm[c]);
        c+=1;
        start=end;
    
    start=0;
    
    for c in range(clz):
        clz_idx = np.where(py==c)[0];
        if len(clz_idx)<1:continue;
        tmp = data[clz_idx];
        ax2.scatter(tmp[:,0],tmp[:,1],marker=mark[c]);        


    for c in range(clz):
        clz_idx = np.where(py2==c)[0];
        if len(clz_idx)<1:continue;
        tmp = data[clz_idx];
        ax3.scatter(tmp[:,0],tmp[:,1],marker=mark[c]);

   
    plt.show();           

if __name__ == '__main__':

    np.random.seed(12121212);

    clz_ds = 300;
    X,Y = get_dataset([(1,1),(1,2),(2,2),(2,1)],clz_ds=clz_ds,sig=0.3, noise=False)
    print(X);
    print(Y);
    # 数值m越小，类别之间的隶属度差异越大，增强了分类效果，强者越强，弱者越弱
    # m相当于模糊边界，越接近1，越像HCM，传统m的范围在[1.5-2.5]之间。
    fcm = Fcm(4,1.5);
    py = fcm.train(X, 100, max_e=0.00001);

    cent,res =simple_km(X, 4);
    py2 = np.zeros((len(X),));
    for idx,cl in enumerate(res):
        py2[cl]=idx;
    
    show_plt(X, clz_ds, py,py2);        
    print(cent);
#     print(res);


    
    pass