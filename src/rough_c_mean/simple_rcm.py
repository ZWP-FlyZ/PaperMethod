# -*- coding: utf-8 -*-
'''
Created on 2019年2月27日

@author: zwp12
'''

'''
简单的rough C-mean 聚类算法实现

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



class Rcm():
    clu=0;# 聚类数量
    th=0.1# 聚类阈值
    w=0.8;# 计算中心时比重
    
    center=None;# 簇中心
    
    def __init__(self,clu,thresdhold=0.1,w=0.8):
        '''
        clu 聚类数量
        threshold： 判断是否为上界元素的距离界限。
        w：计算中心时，下界元素所占比重
        '''
        self.clu=clu;self.th=thresdhold;self.w=w;
    
    def _dis(self,cent,x):
        return np.sqrt(np.sum((cent-x)**2,axis=1)) 
        
    
    def _update(self,data,up_idx,cent):
        clu = self.clu;
        clu_low = [[] for _ in range(clu)];
        clu_edge = [[] for _ in range(clu)];
        dim = data.shape[1];
        sumA = np.zeros((clu,dim));# 下界元素总和
        sumB = np.zeros((clu,dim));# 边界元素总和
        cotA = np.zeros((clu,)); # 下界元素计数
        cotB = np.zeros((clu,));# 边界元素计数
        for jj in up_idx:
            xj = data[jj];
            distance = self._dis(cent, xj);
            ii = np.argmin(distance);# 距离最小的类
            dmin = distance[ii];
            flag=True;# 是否xj判断维下界元素的标志位
            for ci in range(clu):
                if ci == ii:continue; ## 跳过自身比较
                elif abs(dmin - distance[ci])<self.th:
                    # 存在相近的类，将其判断位边界元素
                    flag=False;
                    # 将xj加入到ci类的边界元素中
                    sumB[ci] = sumB[ci]+xj;
                    cotB[ci] += 1;
                    clu_edge[ci].append(jj);
                    
            if(flag):
                ## xj 是下界元素
                sumA[ii] = sumA[ii]+xj;
                cotA[ii] += 1;
                clu_low[ii].append(jj);
            else:
                ## xj 是边界元素
                sumB[ii] = sumB[ii]+xj;
                cotB[ii] += 1;
                clu_edge[ii].append(jj);    
        
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
        return (clu_low,clu_edge),ncent,err,(cotA,cotB);
        
    
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
            res,ncent,err,rec= self._update(data, updata_idx, self.center);
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

#     np.random.seed(12121212);

    clz_ds = 300;
    X,Y = get_dataset([(1,1),(1,2),(2,2),(2,1)],clz_ds=clz_ds,sig=0.24, noise=False)
    print(X);
    print(Y);
    # 数值m越小，类别之间的隶属度差异越大，增强了分类效果，强者越强，弱者越弱
    # m相当于模糊边界，越接近1，越像HCM，传统m的范围在[1.5-2.5]之间。
    rcm = Rcm(4,0.08,0.8);
    py = rcm.train(X, 100, max_e=0.00001);

    cent,res =simple_km(X, 4);
    py2 = np.zeros((len(X),));
    for idx,cl in enumerate(res):
        py2[cl]=idx;
#     
    show_plt(X, clz_ds, py,py2);        
    print(cent);
    print(res);


    
    pass