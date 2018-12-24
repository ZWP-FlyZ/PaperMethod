# -*- coding: utf-8 -*-
'''
Created on 2018年12月12日

@author: zwp
'''
from keras.initializers import glorot_uniform

'''
kreas矩阵分解实现预训练

'''

import tensorflow as tf;
import numpy as np;
import time;
from tensorflow import keras;
from tensorflow.keras.layers import Input,Dense,Multiply,Reshape,Lambda,Concatenate
from tensorflow.keras.initializers import  glorot_uniform,constant
from tensorflow.keras.backend import sum,one_hot;
from tools import SysCheck;


random_seed = 121212;
reg_p = 0.000001;

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';

NoneValue= 0;
# 初始化参数中 正态分布标准差
rou = 0.1;
# 在矩阵分解中 正则化 参数
lamda = 0.04;

# 隐属性数
f = 32;

#训练次数
epoch = 150
# 学习速率
learn_rate = 0.005;
# 
batch_size=1;

spas=[5]

us_shape=(339,5825);
case = 1;
loadvalues=False;
continue_train=True;


def get_model(us_shape=(339,5825),hid_feat=100,mu=0.909):
    '''
    MF_baseline
    '''
    u_in_ = Input((1,),dtype='int32');
    s_in_ = Input((1,),dtype='int32');
    
    u_in = Lambda(lambda x:one_hot(x,us_shape[0]))(u_in_);
    s_in = Lambda(lambda x:one_hot(x,us_shape[1]))(s_in_);
    
    u_in=Reshape((us_shape[0],))(u_in);
    s_in=Reshape((us_shape[1],))(s_in);  
    
    u_hid = Dense(hid_feat,use_bias=True,kernel_regularizer=keras.regularizers.l2(reg_p),
                kernel_initializer=glorot_uniform(seed=random_seed))(u_in);
    s_hid = Dense(hid_feat,use_bias=True,kernel_regularizer=keras.regularizers.l2(reg_p),
                kernel_initializer=glorot_uniform(seed=random_seed))(s_in);
                
#     bu = Dense(1,use_bias=False,kernel_initializer=constant(mu/2))(u_in);
#     bs = Dense(1,use_bias=False,kernel_initializer=constant(mu/2))(s_in);
    
    out = Multiply()([u_hid,s_hid]);
#     out = Concatenate(axis=1)([out,bu,bs]);

    
    out = Lambda(lambda x: sum(x, keepdims=True))(out);
    
    model = keras.models.Model(inputs=[u_in_, s_in_], outputs=out)
    return model

def data_splite(d):
    u = d[:,0].astype(np.int32).reshape(-1,1);
    s = d[:,1].astype(np.int32).reshape(-1,1);
    r = d[:,2].astype(np.float32);
    return u,s,r;

def mf_bl_run(spa,case):
    train_data = base_path+'/Dataset/ws/train_n/sparseness%.1f/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test_n/sparseness%.1f/test%d.txt'%(spa,case);
       
    model_save_path=base_path+'/Dataset/mf_baseline_values/model_spa%.1f_case%d.h5'%(spa,case);
    
    print('开始实验，稀疏度=%.1f,case=%d'%(spa,case));
    print ('加载训练数据开始');
    now = time.time();
    trdata = np.loadtxt(train_data, dtype=float);
    n = np.alen(trdata);
    print ('加载训练数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));
    
    print ('加载测试数据开始');
    tnow = time.time();
    ttrdata = np.loadtxt(test_data, dtype=float);
    n = np.alen(trdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),n));    
    
    print ('清除数据开始');
    tnow = time.time();
    idx = np.where(trdata[:,2]>0);
    train_data=trdata[idx];
    idx = np.where(ttrdata[:,2]>0);
    test_data=ttrdata[idx];
    print ('清除数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    print ('预处理数据开始');
    tnow = time.time();
    train_data = data_splite(train_data);
    test_data = data_splite(test_data);
    print ('预处理数据结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    
    print ('训练模型开始');
    tnow = time.time();
    
    if loadvalues:
        model = keras.models.load_model(model_save_path);
    else:
        model = get_model(hid_feat=f);
    
    if continue_train:
        model.compile(optimizer=keras.optimizers.Adagrad(learn_rate), 
                  loss=keras.losses.mse, 
                  metrics=['mae'])
    
        modelckpit = keras.callbacks.ModelCheckpoint(model_save_path+'/ckpt{epoch:02d}.h5', 
                                                    monitor='val_mae',save_best_only=False);

        ear_stop = keras.callbacks.EarlyStopping(monitor='mean_absolute_error',min_delta=0.005,patience=8);
        
        his = model.fit(x=[train_data[0],train_data[1]],y=train_data[2],
                  batch_size=batch_size, epochs=epoch,
                validation_data = ([test_data[0],test_data[1]],test_data[2]),
#                   callbacks=[ear_stop,modelckpit]
                  );
        print(his.history);
        model.save(model_save_path+'/class127_model.h5');                                                    


def test():
    
    input1 = keras.layers.Input(shape=(1,),name='i1')
    x1 = keras.layers.Dense(8, activation='relu')(input1)
    input2 = keras.layers.Input(shape=(1,),name='i2')
    x2 = keras.layers.Dense(8, activation='relu')(input2)
    added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])
    
    out = keras.layers.Dense(4)(added)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)
    
    model.compile(optimizer=keras.optimizers.Adagrad(learn_rate), 
                  loss=keras.losses.mse, 
                  metrics=['mae']);
    
    a = np.random.uniform(size=(100,1))
    b = np.random.uniform(size=(100,1))
    c = np.random.uniform(size=(100,4))
    model.fit(x={'i1':a,'i2':b},y=c,batch_size=5,epochs=10);
    


if __name__ == '__main__':
#     test();
#     get_model();
    for spa in spas:
        for ca in range(6,13):
            case = ca;
            mf_bl_run(spa,case);
    pass