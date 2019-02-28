# -*- coding: utf-8 -*-
'''
Created on 2019年2月28日

@author: zwp12
'''

'''
简单的rough fuzzy C-mean 聚类算法实现

'''

import numpy as np;
import random;
import matplotlib.pyplot as plt;


def distance_(a,b,tmp_dis_a):
    
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
            dis,tmp_dis_a= distance_(cents, data[i], tmp_dis_a);
            tagk= np.argmin(dis);
            res[tagk].append(i);
        last_c = np.copy(cents);
        for i in range(k):
            cents[i]=np.mean(data[res[i]],axis=0);    
        bout = np.sum(cents-last_c);
        if bout==0:break;

    return cents,res;



class Rfcm():
    clu=0;# 聚类数量
    th=0.1# 聚类阈值
    w=0.8;# 计算中心时比重
    m=2;# 模糊度
    center=None;# 簇中心
    
    def __init__(self,clu,m=2,thresdhold=0.1,w=0.8):
        '''
        clu 聚类数量
        threshold： 判断是否为上界元素的距离界限。
        w：计算中心时，下界元素所占比重
        '''
        self.clu=clu;self.th=thresdhold;
        self.w=w;self.m = m;
    
    def _dis(self,cent,x):
        return np.sqrt(np.sum((cent-x)**2,axis=1)) 
        
    def _cal_u(self,dis):
        # 计算隶属度
        v = dis**(1.0/(self.m-1.0));
        tmp_u = np.zeros((self.clu,));
        for c in range(self.clu):
            # 当前点若就是中心点时需要进行特殊处理
            tt = np.divide(v[c],v,out=np.ones_like(v),where=v!=0);
            tmp_u[c] = np.sum(tt)**(-1);        
        return tmp_u;
    
    def _get_max2_idx(self,uj):
        # 返回最大的两个隶属度的类
        idx = np.argsort(uj);
        return (idx[-1],idx[-2]);    
        
    
    
    def _update(self,data,up_idx,cent):
        clu = self.clu;
        clu_low = [[] for _ in range(clu)];
        clu_edge = [[] for _ in range(clu)];
        
        Ulow = np.zeros((len(up_idx),clu),float);# 下界的隶属度
        Uedge = np.zeros((len(up_idx),clu),float);# 边界的隶属度
        
        dim = data.shape[1];
        sumA = np.zeros((clu,dim));# 下界元素总和
        sumB = np.zeros((clu,dim));# 边界元素总和
        cotA = np.zeros((clu,)); # 下界元素计数
        cotB = np.zeros((clu,));# 边界元素计数
        for jj in up_idx:
            xj = data[jj];
            # 计算距离
            distance = self._dis(cent, xj);
            
            # 计算隶属度 #
            uj = self._cal_u(distance);
            
            # 下界与边界分类 #
            # 计算隶属度最大的两个簇
            ci1,ci2 = self._get_max2_idx(uj);
            if (abs(uj[ci1]-uj[ci2]))>self.th:
                # xj 属于下界元素
                sumA[ci1] = sumA[ci1]+xj;
                cotA[ci1] += 1;
                clu_low[ci1].append(jj);
                # 将下界隶属度最高的类设置为1,其他类的隶属度为0
                Ulow[jj,ci1]=1.0;
                pass;
            else:
                # xj 属于边界元素，将xj加入到ci1，ci2两个簇的边界中
                clu_edge[ci1].append(jj);
                clu_edge[ci2].append(jj);
                Uedge[jj]=uj; # xj更新边界隶属度，xj的下界隶属度不变
                tt = uj[ci1]**self.m;
                sumB[ci1] = sumB[ci1]+xj*tt;
                cotB[ci1] += tt;
                tt = uj[ci2]**self.m;
                sumB[ci2] = sumB[ci2]+xj*tt;
                cotB[ci2] += tt;                
                
                pass;
                    
        # 更新中心点
        ncent = np.zeros_like(cent);
        for ci in range(clu):
            if cotA[ci]==0:
                ncent[ci]=sumB[ci]/cotB[ci];
            elif cotB[ci]==0:
                ncent[ci]=sumA[ci]/cotA[ci];
            else:
                ncent[ci]=self.w * sumA[ci]/cotA[ci] + \
                (1-self.w) * sumB[ci]/cotB[ci];
        
        err = np.sum(np.abs(ncent-cent));
        cotB = [];
        for c in clu_edge:
            cotB.append(len(c));
        return (clu_low,clu_edge),ncent,err,(cotA,cotB),(Ulow,Uedge);
        
    
    def train(self,data,max_loop=100,max_e = 0.00001):
        ds = len(data);
        epo = 1;# 迭代次数
        updata_idx = np.arange(ds,dtype=np.int);

        # 初始化中心点
        eidx = np.random.choice(updata_idx,size=self.clu,
                                    replace=False);
        self.center = data[eidx];
        
        while epo<=max_loop:
            np.random.shuffle(updata_idx);# 随机训练顺序
            res,ncent,err,rec,memU= self._update(data, updata_idx, self.center);
            print('epoch=%d \terr=%.6f \tclu_ele_rec=%s,%s'%(epo,err,str(rec[0]),str(rec[1])));
            print();
            self.center=ncent;
            if err<=max_e:break;
            epo+=1;
        return res;
#############################################  end ##################################    
        
    
    

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
    
    clu_low,clu_edge = py;
    clu_low=np.array(clu_low);
    clu_edge=np.array(clu_edge);
    for c in range(clz):
        tmp = data[clu_low[c]];
        ax2.scatter(tmp[:,0],tmp[:,1],marker=mark[c]);        
    
    for c in range(clz):
        tmp = data[clu_edge[c]];
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
    X,Y = get_dataset([(1,1),(1,2),(2,2),(2,1)],clz_ds=clz_ds,sig=0.27, noise=False)
    print(X);
    print(Y);
    # 数值m越小，类别之间的隶属度差异越大，增强了分类效果，强者越强，弱者越弱
    # m相当于模糊边界，越接近1，越像HCM，传统m的范围在[1.5-2.5]之间。
    rcm = Rfcm(4,2.0,0.1,0.9);
    py = rcm.train(X, 100, max_e=0.00001);

    cent,res =simple_km(X, 4);
    py2 = np.zeros((len(X),));
    for idx,cl in enumerate(res):
        py2[cl]=idx;
#     
    show_plt(X, clz_ds, py,py2);        
    print(cent);
    print(res);




if __name__ == '__main__':
    pass