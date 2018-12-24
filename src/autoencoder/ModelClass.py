# -*- coding: utf-8 -*-
'''
Created on 2018年12月13日

@author: zwp
'''


import numpy as np;
import time;
import math;
import os;
from tools import SysCheck;
import multiprocessing as mp;


class MF_bl_ana:
    us_shape=None;
    size_f = None;
    mean = None;
    values = None;
    ana = None;
    def __init__(self,us_shape,size_f,mean):
        self.us_shape = us_shape;
        self.size_f = size_f;
        self.mean = mean;
        self.ana = np.zeros(us_shape);
        self.values={
            'P':np.random.uniform(-0.3,0.3,(us_shape[0],size_f))/np.sqrt(size_f),
            'Q':np.random.uniform(-0.3,0.3,(us_shape[1],size_f))/np.sqrt(size_f),
#             'P':np.random.rand(us_shape[0],size_f)/np.sqrt(size_f),
#             'Q':np.random.rand(us_shape[1],size_f)/np.sqrt(size_f),
            'bu':np.zeros(us_shape[0]),
            'bi':np.zeros(us_shape[1])           
        };
        
    def predict(self,u,i):
        P = self.values['P'];
        Q = self.values['Q'];
        bu = self.values['bu'];
        bi = self.values['bi'];
        res=self.mean+bu[u]+bi[i];
        res += np.sum(P[u]*Q[i]);
        return res;
    
    def predictAll(self,u,i):
        P = self.values['P'];
        Q = self.values['Q'];
        bu = self.values['bu'];
        bi = self.values['bi'];
        res=self.mean+bu[u]+bi[i];
        res += np.sum(P[u]*Q[i],axis=1);
        return res;
    
    def value_optimize(self,u,i,rt,lr,lamda):
        P = self.values['P'];
        Q = self.values['Q'];
        bu = self.values['bu'];
        bi = self.values['bi'];
        pt =  self.predict(u, i);       
        eui = rt-pt;
        # 更改baseline 偏置项
        bu[u] += lr * (eui-lamda*bu[u]); 
        bi[i] += lr * (eui-lamda*bi[i]);
        #更改MF 参数
        tmp = lr * (eui*Q[i]-lamda*P[u]);
        Q[i] += lr * (eui*P[u]-lamda*Q[i]);
        P[u]+=tmp;
        self.values['P']=P;
        self.values['Q']=Q;
        self.values['bu']=bu;
        self.values['bi']=bi;
        return pt;
        
    def train_mat(self,x,y,val=None,repeat=1,learn_rate=0.01,lamda=0.02,save_path=None):
        print('|-->训练开始，learn_rate=%f,repeat=%d'%(learn_rate,repeat));
        now = time.time();
        d_s = len(x);
        idx_li = np.arange(d_s,dtype=np.int);
#         ana = self.ana;
        for rep in range(repeat):
            tnow=time.time();
            maeAll=0.0;rmseAll=0.0;
            np.random.shuffle(idx_li);
            for idx in idx_li:
                rt = y[idx];
                u = x[idx,0];
                s = x[idx,1];
                pt = self.value_optimize(u, s, rt, learn_rate,lamda);
                t = abs(rt-pt);
#                 ana[u, s]=t;
                maeAll+=t;
            maeAll = maeAll / d_s;          
            if save_path != None and False:
                self.saveValues(save_path);
            print('|---->step%d 耗时%.2f秒 MAE=%.6f RMSE=%.6f|'%(rep+1,(time.time()-tnow),maeAll,rmseAll));
            
            if val!= None and rep!=0 and rep%5==0:
                val_u = val[0][:,0];
                val_s = val[0][:,1];
                val_y = val[1];
                val_py = self.predictAll(val_u, val_s);
                val_mae = np.mean(np.abs(val_y-val_py))
                print('val_mae = %.6f'%val_mae);
            
#         list_ana = self.ana.reshape((-1,));    
#         ind = np.argsort(-list_ana)[:1000];
#         ana_sorted = list_ana[ind];
#         arg_list = [[int(i/shp[1]),int(i%shp[1])]for i in ind];
#         ori_list = [R[i[0],i[1]] for i in arg_list];
#         if not os.path.isdir(save_path):
#             os.mkdir(save_path);
#         np.savetxt(save_path+'/ana_value.txt',np.array(ana_sorted),'%.6f');
#         np.savetxt(save_path+'/ana_ind.txt',np.array(arg_list),'%d');
#         np.savetxt(save_path+'/ana_ori_value.txt',np.array(ori_list),'%.6f');
        print('|-->训练结束，总耗时%.2f秒  learn_rate=%.3f,repeat=%d \n'%((time.time()-now),learn_rate,repeat));


    def preloadValues(self,path):
        if os.path.exists(path+'/Q.txt'):
            self.values['Q']=np.loadtxt(path+'/Q.txt', float);
        if os.path.exists(path+'/P.txt'):
            self.values['P']=np.loadtxt(path+'/P.txt', float);        
        if os.path.exists(path+'/bu.txt'):
            self.values['bu']=np.loadtxt(path+'/bu.txt', float);        
        if os.path.exists(path+'/bi.txt'):
            self.values['bi']=np.loadtxt(path+'/bi.txt', float);
           
    def saveValues(self,path):
        if not os.path.isdir(path):
            os.mkdir(path);
        np.savetxt(path+'/P.txt',self.values['P'],'%.6f');
        np.savetxt(path+'/Q.txt',self.values['Q'],'%.6f');
        np.savetxt(path+'/bu.txt',self.values['bu'],'%.6f');
        np.savetxt(path+'/bi.txt',self.values['bi'],'%.6f');
  
    def exisValues(self,path,isUAE=True):
        if not os.path.exists(path+'/Q.txt'):
            return False
        if not os.path.exists(path+'/P.txt'):
            return False        
        if not os.path.exists(path+'/bu.txt'):
            return False        
        if not os.path.exists(path+'/bi.txt'):
            return False       
        return True;
    def getValues(self):
        return self.values;



class DenoiseAutoEncoder:
    
    def __init__(self,X_n,hidden_n,actfun1,deactfun1,actfun2,deactfun2,check_none=None):
        self.size_x = X_n;
        self.size_hidden=hidden_n;
        self.func1 = actfun1;
        self.defunc1 =deactfun1;
        self.func2 = actfun2;
        self.defunc2 =deactfun2;
        self.check_none= check_none;
        np.random.seed(1212121);
        self.values= {
            'w1':np.random.uniform(-0.4,0.4,(self.size_x,self.size_hidden))/np.sqrt(hidden_n),
            'w2':np.random.uniform(-0.4,0.4,(self.size_hidden,self.size_x))/np.sqrt(hidden_n),
            'b1':np.random.uniform(-0.4,0.4,(1,self.size_hidden))/np.sqrt(hidden_n),
            'b2':np.random.uniform(-0.4,0.4,(1,self.size_x))/np.sqrt(hidden_n),
            'h':None
            };
    
    ## 注意这里计算过方法必须保证NoneValue=0
    def calculate(self,x,save_h=True):
        xsp1 = x.shape[0];
        x = np.reshape(x, (xsp1,self.size_x));
        h = self.func1(np.matmul(x,self.values['w1'])+self.values['b1']);
        if save_h:self.values['h']=h;
        y = self.func2(np.matmul(h,self.values['w2'])+self.values['b2']);
        return y;

    def calFill(self,R,x_axis=1):
        '''
        R中与自编码器输入x的x_size对应轴号
        '''
        tR = R;
        if x_axis==0:
            tR = R.T;  
        return self.calculate(tR,False);

    
    def evel(self,py,y,mask_value=0):
        '''
        假定py已经去除mask项
        '''
        index = np.where(y!=mask_value);
        if len(index[0])==0:return 0,0;
        delta = np.abs(py[index]-y[index]);
        mae = np.mean(delta);
#         rmse = np.mean(delta**2);
        rmse=0.0;
        return mae,math.sqrt(rmse);
    
    # 更新参数
    def layer_optimize(self,py,y,oriy,
                       learn_rate,# 学习速率
                       mask_value=0,
                       err_weight=1.0
                       ):
        b1 = self.values['b1'];
        w1 = self.values['w1'];
        b2 = self.values['b2'];
        w2 = self.values['w2'];
        h  = self.values['h'];
        lr = learn_rate;
        
        #替换py中无效的值为mask_value;
        py = np.where(y!=mask_value,py,y);
        
        origin_w2 = w2.copy();
        # 输出层的调整
        gjs = err_weight*(py-y)*self.defunc2(py);# 输出层中的梯度
        tmp = gjs*lr;
        b2 = b2 - tmp; # 调整b2
        
        deltaW = np.matmul(
            np.reshape(h,(self.size_hidden,1)),# 隐层输出
            np.reshape(tmp, (1,self.size_x))
            );
        w2 = w2 - deltaW;# 调整w2
        
#         tmp = origin_w2* gjs;
#         tmp =np.sum(tmp,axis=1);
#         tmp = np.matmul(gjs,origin_w2.T);
        tmp = np.matmul(gjs,origin_w2.T);
        gis = tmp*self.defunc1(h);
        
        tmp = gis* lr;
        
        b1 = b1 - tmp;# 更新b1

        deltaW = np.matmul(
            np.reshape(oriy,(self.size_x,1)),# 输入层
            np.reshape(tmp, (1,self.size_hidden))
            );
        w1 = w1 - deltaW;# 调整w1
        

        self.values['b2']=b2;
        self.values['w2']=w2;        
        self.values['b1']=b1;
        self.values['w1']=w1;
                        

    def train(self,X,extX,valR=None,learn_param=[0.01,1,1.0],repeat=1,save_path=None,mask_value=0,weight_list=None):
        '''
        注意输入X为一个矩阵(batch,x_size)
        extX为假设原数据集
        '''
        self.lp=learn_param;
        lr = learn_param[0];
        de_repeat = learn_param[1];
        de_rate = learn_param[2];
        print('-->训练开始，learn_param=',self.lp,'repeat=%d \n'%(repeat));
        now = time.time();
        shape1=extX.shape[0];
        dataidx = np.arange(shape1,dtype=np.int);
        ae_val_res=[];
        val_size = np.count_nonzero(valR);
        for rep in range(repeat):
            tnow=time.time();
            maeAll=0.0;rmseAll=0.0;
            np.random.shuffle(dataidx);
            for i in dataidx:
                
                x = X[i:i+1,:];
                extx = extX[i:i+1,:];
                py = self.calculate(x);

                mae,rmse=self.evel(py, x,mask_value);
                err_weight=1.0;
                if weight_list is not None:
                    err_weight = weight_list[i];
                self.layer_optimize(py,x,extx,learn_rate=lr,err_weight=err_weight);
                maeAll+=mae/shape1;
                rmseAll+=rmse/shape1;
#             print(py);
            if rep>0 and rep%de_repeat==0:
                lr*=de_rate;
            if save_path != None:
                self.saveValues(save_path);
            print('---->step%d 耗时%.2f秒 MAE=%.6f RMSE=%.6f'%(rep+1,(time.time()-tnow),maeAll,rmseAll));
        
            if valR is not None and rep!=0 and rep%5==0:
                pR = self.calFill(X);
                delta = np.subtract(pR,valR,out=np.zeros_like(pR),where=valR>0);
                valmae = np.sum(np.abs(delta))/val_size;
                ae_val_res.append(valmae);
                print('mae=%.6f'%(valmae*20));
            
        
        print('\n-->训练结束，总耗时%.2f秒 ,repeat=%d \n'%((time.time()-now),repeat));
        return ae_val_res;
        
    def preloadValues(self,path,isUAE=True):
        if os.path.exists(path+'/w1_%s.txt'%(isUAE)):
            self.values['w1']=np.loadtxt(path+'/w1_%s.txt'%(isUAE), np.float64);
        if os.path.exists(path+'/w2_%s.txt'%(isUAE)):
            self.values['w2']=np.loadtxt(path+'/w2_%s.txt'%(isUAE), np.float64);        
        if os.path.exists(path+'/b1_%s.txt'%(isUAE)):
            self.values['b1']=np.loadtxt(path+'/b1_%s.txt'%(isUAE), np.float64).reshape(1,self.size_hidden);        
        if os.path.exists(path+'/b2_%s.txt'%(isUAE)):
            self.values['b2']=np.loadtxt(path+'/b2_%s.txt'%(isUAE), np.float64).reshape(1,self.size_x);
        if os.path.exists(path+'/h_%s.txt'%(isUAE)):
            self.values['h']=np.loadtxt(path+'/h_%s.txt'%(isUAE), np.float64);            
    def saveValues(self,path,isUAE=True):
        if not os.path.isdir(path):
            os.makedirs(path);
        np.savetxt(path+'/w1_%s.txt'%(isUAE),self.values['w1'],'%.12f');
        np.savetxt(path+'/w2_%s.txt'%(isUAE),self.values['w2'],'%.12f');
        np.savetxt(path+'/b1_%s.txt'%(isUAE),self.values['b1'],'%.12f');
        np.savetxt(path+'/b2_%s.txt'%(isUAE),self.values['b2'],'%.12f');
        np.savetxt(path+'/h_%s.txt'%(isUAE),self.values['h'],'%.12f');    
    def exisValues(self,path,isUAE=True):
        if not os.path.exists(path+'/w1_%s.txt'%(isUAE)):
            return False;
        if not os.path.exists(path+'/w2_%s.txt'%(isUAE)):
            return False;        
        if not os.path.exists(path+'/b1_%s.txt'%(isUAE)):
            return False;      
        if not os.path.exists(path+'/b2_%s.txt'%(isUAE)):
            return False;
        if not os.path.exists(path+'/h_%s.txt'%(isUAE)):
            return False;        
        return True;
    def getValues(self):
        return self.values;    
   
############################ end class ###########################

def f(R,start,end):
    y_s =  R.shape[1];
    tR =R.T;
    W = np.zeros((y_s,y_s));
    for i in range(start,end):
        a = tR[i];
        if end>=5825 and i%60 ==0:
            print('step%d scf_w'%i);
        for j in range(i+1,y_s):
            b = tR[j];
            log_and = (a!=0) & (b!=0);
            ws = np.zeros_like(a);
            ws+=np.subtract(a,b,out=np.zeros_like(a),where=log_and)
            ws=np.sum(ws**2);
            W[i,j]=W[j,i]= 1.0/np.exp(np.sqrt(ws));  
    return W;

class CF():
    UW = None;
    US = None;
    SW = None;
    SS = None;
    
    feat_w_us = None;
    feat_w_su = None;
    
    def __init__(self,shape,mode):
        self.shape=shape;
        self.mode=mode;
        if mode == 1:
            self.UW = np.zeros((shape[0],shape[0]));
        elif mode == 2:
            self.SW = None;
        elif mode == 3:
            self.UW = np.zeros((shape[0],shape[0]));
            self.SW = None;
    
    def train(self,R,oriR=None,k=10):
        mode = self.mode
        if mode==1:
            self.ucf_w(R);
            self.ucf_S(k);
        elif mode==2:
            self.scf_w(R);
            self.scf_S(k);
        elif mode==3:
            self.ucf_w(R[0]);
            self.ucf_S(k[0]);
            self.scf_w(R[1]);
            self.scf_S(k[1]);
        
    def evel(self,valR,R,u_s_rate=0.5):
        valu,vals = np.where(valR>0);
        y = valR[valu,vals];
        ds = len(valu);
        py = np.zeros_like(y);
        mode = self.mode;
        tmp=0;
        for i in range(ds):
            if mode==1:
                tmp=self.pre_u(valu[i],vals[i],R);
            elif mode == 2:
                tmp=self.pre_s(valu[i],vals[i],R);
            elif mode ==3:
                tmp=u_s_rate*self.pre_u(valu[i],vals[i],R[0]);
                tmp+=(1-u_s_rate)*self.pre_s(valu[i],vals[i],R[1]);
            py[i]=  tmp;
        
        mae = np.abs(py-y);
        nmae = np.sum(mae)/np.sum(y);
        mae = np.mean(mae);
        return mae,nmae;
          
        
    def ucf_w(self,R):
        x_size,y_size = R.shape;
        W = self.UW;
        for i in range(x_size):
            a = R[i];
            if i%60 ==0:
                print('step%d ucf_w'%i);
            for j in range(i+1,x_size):
                b = R[j];
                log_and = (a!=0) & (b!=0);
                ws = np.zeros_like(a);
                ws+=np.subtract(a,b,out=np.zeros_like(a),where=log_and)
                ws=np.sum(ws**2);
                W[i,j]=W[j,i]= 1.0/math.exp(np.sqrt(ws));



    def scf_w(self,R,useP=True):
        x_size,y_size = R.shape;
        sps = [376,404,439,486,551,654,853]
        res = [];
        start=0;
        if useP:
            pool = mp.Pool(8);
            for i in range(7):
                end = min(start+sps[i],5825);
                res.append(pool.apply_async(f,(R,start,end)));
                start=end;
            res.append(pool.apply_async(f,(R,start,5825)));
        else:
            pool = mp.Pool(1);
            res.append(pool.apply_async(f,(R,start,5825)));
        pool.close();
        pool.join();
        
        ws = [];
        for it in res:
            ws.append(it.get())
#         print(ws);
        W = ws[0];
        for i in range(1,len(ws)):
            W = W + ws[i];
        self.SW = W;
        
    def ucf_S(self,k):
        W = self.UW;
        S = np.argsort(-W, axis=1)[:,0:k];
#         print(W[0,S[0]])
        self.US = S;
    
    def scf_S(self,k):
        W = self.SW;
        S = np.argsort(-W, axis=1)[:,0:k];
        self.SS = S;

    def pre_u(self,u,s,R):
        W = self.UW;
        S = self.US;
        uw = W[u];
        uS = S[u];
        sum_up=0;sum_down=0;
        for simu in uS:    
            if uw[simu]<=0.0:
                break;
            if R[simu,s]==0:
                continue;
            rw = uw[simu];            
            sum_up+= rw*R[simu,s];
            sum_down+=rw;
        if sum_down != 0:
            return sum_up/sum_down;
        else:
            return 0.2;

    def pre_s(self,u,s,R):
        W = self.SW;
        S = self.SS;
        sS = S[s];
        sum_up=0;sum_down=0;
        for sims in sS:    
            if W[s,sims]<=0.0:
                break;
            if R[u,sims]==0:
                continue;
            rw = W[s,sims];            
            sum_up+= rw*R[u,sims];
            sum_down+=rw;
        if sum_down != 0:
            return sum_up/sum_down;
        else:
            return 0.2;



