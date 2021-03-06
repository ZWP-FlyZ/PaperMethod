# -*- coding: utf-8 -*-
'''
Created on 2019年2月28日

@author: zwp
'''




'''

利用RFCM进行聚类
'''


import numpy as np;
import time;
import random;
import os;
from tools import SysCheck;
from tools import localload;


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work';
origin_path = base_path+'/Dataset/ws/rtmatrix.txt';
ser_info_path=base_path+'/Dataset/ws/localinfo/ws_info.txt';
ser_info_more_path=base_path+'/Dataset/ws/localinfo/ws_info_more.txt';
fcm_w_out = base_path+'/Dataset/ws/localinfo/service_rfcm_w.txt';





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
    data[:,2]=data[:,2]/200.0;
    data[:,3]=data[:,3]/200.0;
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
    
    def _update_U_imped(self,data,uidx,cent,di=1):
        '''
        > 改进的模糊c运算方法
        '''
        m = self.m;
        clu = self.clu;
        clu_rec=[0]*clu;
        ds = len(uidx);
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
            
            # 调节最小距离，避免接近中心的数据点权重过大
            vidx = np.where(v<0.2);
            v[vidx]=0.2;
            v = (1.0/v)**(1.0/(m-1.0));
            for c in range(clu):
                tt = v[c]**m;
                new_cent[c] += xj*tt;
                new_cent_dw[c] += tt;
                
            max_idx = np.argmax(v);
            clu_result[max_idx].append(idx);
            clu_rec[max_idx] = clu_rec[max_idx]+1;
            self.U[idx]= v;
        
        # 计算隶属度矩阵U
        self.U[idx] = ds * self.U[idx] / np.sum(self.U[idx]);
    
        # 计算新中心
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
        data[:,2]=data[:,2]/100.0;
        data[:,3]=data[:,3]/100.0;


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

class Rfcm():
    clu=0;# 聚类数量
    th=0.1# 聚类阈值
    w=0.8;# 计算中心时比重
    m=2;# 模糊度
    center=None;# 簇中心
    memU=None;# 隶属度对(Ulow,Uedge)
    
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
        
    
    
    def _update(self,data,up_idx,cent,di=1):
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
            dis_lg = haversine(cent[:,0],cent[:,1],xj[0],xj[1],r=di);
            dis_lg = np.reshape(dis_lg,[-1,1])**2;
            dis_oth = self._dis(cent[:,2:],xj[2:]).reshape(-1,1)**2;            
            
            v = np.concatenate((dis_lg,dis_oth),axis=1);
            distance= np.sum(v,axis=1);
            
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
        
    
    def train(self,data,max_loop=100,max_e = 0.00001,di=1):
        ds = len(data);
        epo = 1;# 迭代次数
        updata_idx = np.arange(ds,dtype=np.int);

        # 格式化数据
        data[:,2]=data[:,2]/150.0;
        data[:,3]=data[:,3]/150.0;

        # 初始化中心点
        eidx = np.random.choice(updata_idx,size=self.clu,
                                    replace=False);
        self.center = data[eidx];
        
        while epo<=max_loop:
            np.random.shuffle(updata_idx);# 随机训练顺序
            res,ncent,err,rec,self.memU= self._update(data, updata_idx, self.center,di);
            print('epoch=%d \terr=%.6f \tclu_ele_rec=%s,%s'%(epo,err,str(rec[0]),str(rec[1])));
            print();
            self.center=ncent;
            if err<=max_e:break;
            epo+=1;
        return ncent,res;
#############################################  end ##################################


def run():
    
    ser_loc = localload.load(ser_info_path);
    ser_loc_m = localload.load_locmore(ser_info_more_path);
    R = np.loadtxt(origin_path,np.float);
    
    if os.path.isfile(fcm_w_out):
        os.remove(fcm_w_out);

    idx = np.where(R<0);
    R[idx]=0;

    service_sum = np.sum(R,axis=0);
    service_cot = np.count_nonzero(R, axis=0);
    service_mean = np.divide(service_sum,service_cot,
        out=np.zeros_like(service_sum),where=service_cot!=0);
    all_mean = np.sum(service_sum)/np.sum(service_cot);
    service_mean[np.where(service_cot==0)] = all_mean;

    
    data=[];
    names=[];
    area=[];
    k=6;
    for sid in range(5825):
        sn = ser_loc[sid][1];
        names.append(sn);
        area.append(ser_loc_m[sn][0])
        lc = [];
        lc.extend(ser_loc_m[sn][1]);
        # 添加ip
        lc.extend(ser_loc[sid][2][:2]);
        lc.append(service_mean[sid]);

        data.append(lc);
    data=np.array(data);

    rfcm = Rfcm(k,m=1.1,thresdhold=0.1,w=0.85);
    cent,res = rfcm.train(data, max_loop=100, max_e=0.00001,di=1);
    
    print(cent);
    print(res[0]);
    print(res[1]);
    
    np.savetxt(fcm_w_out, np.concatenate(rfcm.memU,axis=1), '%.6f');
    
    print('######################## low ############### ')
    for i in range(k):
        tmp=[];
        tmp2=[];
        for id in res[0][i]:
            if names[id] not in tmp2:
                tmp2.append(names[id]);
                tmp.append(area[id]);
        print(tmp)
        print(tmp2);
        print();
    
    print('######################## edge ############### ')

    for i in range(k):
        tmp=[];
        tmp2=[];
        for id in res[1][i]:
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