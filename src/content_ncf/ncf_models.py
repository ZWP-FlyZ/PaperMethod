# -*- coding: utf-8 -*-
'''
Created on 2019年1月7日

@author: zwp
'''

import tensorflow as tf;
import time;
from tools.ncf_tools import reoge_data,reoge_data_for_keras
from tools.fwrite import fwrite_append;

# from content_ncf.ncf_param import NcfTraParm,NcfCreParam;
import numpy as np;
import time;

from tensorflow.python import keras;

from tensorflow.python.keras import backend as K;
from tensorflow.python.keras.layers import Input,Lambda,Dense,Concatenate,Dropout
from tensorflow.python.keras.initializers import glorot_uniform, zero, Constant
from tensorflow.python.keras.regularizers import l2;
from tensorflow.python.keras.optimizers import Adagrad;
from tensorflow.python.keras import Model;
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.engine import Layer
from tools.fwrite import fwrite_append

class LossHistory(keras.callbacks.Callback):
    def __init__(self,rec_path,moreinfo=None):
        self.path = rec_path;
        self.info = moreinfo;
    
    def on_epoch_end(self, epo, logs={}):
        val_mae = logs['val_mean_absolute_error'];
        out_s = 'epoch=%d mae=%.6f time=%s'%(epo+1,val_mae,time.asctime());
        print('----->'+out_s);
        fwrite_append(self.path,out_s);
        
    
    
class InvokeRecLayer(Layer):
    def __init__(self, hid_dim,invoked_rec,**kwargs):
        self.output_dim = hid_dim;
        self.ivk = invoked_rec;
        self.sNum = invoked_rec.get_shape().as_list()[1];
        super(InvokeRecLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='ivk_ker', 
                                      shape=(self.sNum, self.output_dim),
                                      initializer=glorot_uniform,
                                      dtype='float32',
                                      trainable=True);
        print(self.kernel);
        super(InvokeRecLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        choosed = K.gather(self.ivk, K.cast(K.squeeze(x, 1),'int32'));
#         choosed=K.cast(choosed,'float32');
        print(choosed);
        return K.dot(choosed,self.kernel);

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)    


class BaseLineLayer(Layer):
    def __init__(self, mean,**kwargs):
        self.output_dim = 1;
        self.mean = mean;
        super(BaseLineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
               
        self.b = self.add_weight(name='B', 
                                      shape=(len(self.mean), self.output_dim),
                                      initializer=zero,
                                      dtype='float32',
                                      trainable=True);                                      
        super(BaseLineLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, u):
        choosed = K.gather(self.b, K.cast(K.squeeze(u, 1),'int32'));
        print(choosed);
        return choosed;

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)






class HidFeatLayer(Layer):
    def __init__(self, in_dim,out_dim,**kwargs):
        self.output_dim = out_dim;
        self.in_dim = in_dim;
        super(HidFeatLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
               
        self.ker = self.add_weight(name='hid_feat', 
                                      shape=(self.in_dim, self.output_dim),
                                      initializer=glorot_uniform(seed=self.in_dim+123456),
                                      dtype='float32',
                                      trainable=True);                                      
        super(HidFeatLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        choosed = K.gather(self.ker, K.cast(K.squeeze(x, 1),'int32'));
        print(choosed);
        return choosed;

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class simple_ncf():
    '''
    由keras实现的ncf模型
    '''
    
    # 用户服务数量
    uNum=None;
    sNum=None;
    
    # 创建参数
    creParm = None;

    # ncf 模型
    model = None;
    
    def __init__(self,NcfCreParam):
        self.uNum,self.sNum=NcfCreParam.us_shape;
        self.creParm = NcfCreParam;
        self.model = self._get_model();
        pass;    
    
    def _get_model(self,):
        
        # 潜在特征
        hid_f = self.creParm.hid_feat;
        # 网络单元数
        units = self.creParm.hid_units;
        reg_p = self.creParm.reg_p;
        drop_p = self.creParm.drop_p;

        
        input_u = Input(shape=(1,),dtype="int32");
        input_s = Input(shape=(1,),dtype="int32");
        print(input_u);
        # one hot 转换
        u_hid = HidFeatLayer(self.uNum,hid_f)(input_u);
        s_hid = HidFeatLayer(self.sNum,hid_f)(input_s);
        print(u_hid,s_hid);
        # 连接               
        out = Concatenate()([u_hid,s_hid]);
        print(out);
        
        # 若干全连接层和dropout
        for idx,unit in enumerate(units):
            out = Dense(unit,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                kernel_regularizer=l2(reg_p))(out)
            out = Dropout(drop_p)(out);
        
        print(out);
        
        out = Dense(1,
            activation=keras.activations.relu,
            kernel_initializer= glorot_uniform(),
            kernel_regularizer=l2(reg_p))(out);            
        print(out);
        return Model(inputs=[input_u,input_s],outputs=out);
        
    def train(self,tp):
        train_x,train_y = reoge_data_for_keras(tp.train_data);
        test_data = reoge_data_for_keras(tp.test_data);
        
        bs = tp.batchsize;# batch_size
        lr = tp.learn_rate;# 学习律
        epoch = tp.epoch;# 迭代次数
        need_load = tp.load_cache_rec;# 是否加载记录数据
        load_path = tp.cache_rec_path;# 缓存路径
        result_path =tp.result_file_path;# 结果文件
        
        ncf_model = self.model;
        
        # 回调处理
        # 日志回调
        myhis = LossHistory(result_path);
        
        
        # early stop回调
        ear_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error',
                            min_delta=0.0002,patience=10); 
        #save回调
        chkpoint = ModelCheckpoint(load_path, monitor='val_mean_absolute_error', 
                                   save_best_only=False,
                                   save_weights_only=True,  
                                   mode='auto');
                                   
        rr = keras.callbacks.ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=2, 
                               verbose=1, mode='auto', 
                               cooldown=3, min_lr=0.0001)                           
        
                                   
        ncf_model.compile(optimizer=Adagrad(lr), 
              loss='mae', 
              metrics=['mae']);
        
        his = ncf_model.fit(train_x,train_y,
                      batch_size=bs,epochs=epoch,
                      validation_data = test_data,
                      callbacks=[chkpoint,myhis,ear_stop],
                      verbose=0);
        val_rec = his.history['val_mean_absolute_error'];
        return min(val_rec),0;
        pass;
                           
class simple_ncf_pp():
    '''
    由keras实现的ncf_pp模型
    '''
    
    # 用户服务数量
    uNum=None;
    sNum=None;
    
    # 创建参数
    creParm = None;

    # ncf 模型
    model = None;
    
    def __init__(self,NcfCreParam):
        self.uNum,self.sNum=NcfCreParam.us_shape;
        self.creParm = NcfCreParam;
        pass;    
    
    def _get_model(self,us_invoked,umean,smean):
        
        # 潜在特征
        hid_f = self.creParm.hid_feat;
        # 网络单元数
        units = self.creParm.hid_units;
        reg_p = self.creParm.reg_p;
        drop_p = self.creParm.drop_p;
        
        input_u = Input(shape=(1,),dtype="int32");
        input_s = Input(shape=(1,),dtype="int32");
        print(input_u);
        # one hot 转换
        u_hid = HidFeatLayer(self.uNum,hid_f)(input_u);
        s_hid = HidFeatLayer(self.sNum,hid_f)(input_s);
        print(u_hid,s_hid);
        
        # 获取调用记录权重
        u_rec = InvokeRecLayer(16,us_invoked)(input_u);
#         s_rec = InvokeRecLayer(8,K.transpose(us_invoked))(input_s);
        
        # baseline 
        bu = BaseLineLayer(umean)(input_u);
        bs = BaseLineLayer(smean)(input_s);
        # 连接               
        out = Concatenate()([u_hid,u_rec,s_hid]);
        print(out);
        
        # 若干全连接层和dropout
        for idx,unit in enumerate(units):
            out = Dense(unit,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                kernel_regularizer=l2(reg_p))(out)
            out = Dropout(drop_p)(out);
        
#         out = Concatenate()([bu,out,bs]);
        print(out);
        
        out = Dense(1,
            activation=keras.activations.relu,
            kernel_initializer= glorot_uniform(),
            kernel_regularizer=l2(reg_p))(out);
                    
        print(out);
        return Model(inputs=[input_u,input_s],outputs=out);
        
    def train(self,tp):
        train_x,train_y = reoge_data_for_keras(tp.train_data);
        test_data = reoge_data_for_keras(tp.test_data);
        
        bs = tp.batchsize;# batch_size
        lr = tp.learn_rate;# 学习律
        epoch = tp.epoch;# 迭代次数
        need_load = tp.load_cache_rec;# 是否加载记录数据
        load_path = tp.cache_rec_path;# 缓存路径
        result_path =tp.result_file_path;# 结果文件
        invoked_rec = K.constant(tp.us_invked,dtype='float32');
        
        umean=tp.umean;
        smean=tp.smean;
        
        ncf_model = self._get_model(invoked_rec,umean,smean);
        self.model = ncf_model;
        # 回调处理
        # 日志回调
        myhis = LossHistory(result_path);
        
        # early stop回调
        ear_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error',
                            min_delta=0.0002,patience=5); 
        
        #save回调
        chkpoint = ModelCheckpoint(load_path, monitor='val_mean_absolute_error', 
                                   save_best_only=False,
                                   save_weights_only=True,  
                                   mode='auto');
        
        rr = keras.callbacks.ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=1, 
                               verbose=1, mode='auto', 
                               cooldown=2, min_lr=0.0001)  
        
        tb = keras.callbacks.TensorBoard(log_dir='/home/zwp/work/tensorboard');
        
        ncf_model.compile(optimizer=keras.optimizers.Adagrad(lr), 
              loss='mae', 
              metrics=['mae']);
        
        his = ncf_model.fit(train_x,train_y,
                      batch_size=bs,epochs=epoch,
                      validation_data = test_data,
                      callbacks=[chkpoint,myhis,ear_stop,tb],verbose=2);
        val_rec = his.history['val_mean_absolute_error'];
        return min(val_rec),0;
        pass;    
        
class simple_ncf_bl():
    '''
    由keras实现的ncf模型
    '''
    
    # 用户服务数量
    uNum=None;
    sNum=None;
    
    # 创建参数
    creParm = None;

    # ncf 模型
    model = None;
    
    def __init__(self,NcfCreParam):
        self.uNum,self.sNum=NcfCreParam.us_shape;
        self.creParm = NcfCreParam;
        
        pass;    
    
    def _get_model(self,umean,smean):
        
        # 潜在特征
        hid_f = self.creParm.hid_feat;
        # 网络单元数
        units = self.creParm.hid_units;
        reg_p = self.creParm.reg_p;
        drop_p = self.creParm.drop_p;

        input_u = Input(shape=(1,),dtype="int32");
        input_s = Input(shape=(1,),dtype="int32");
        print(input_u);
        # one hot 转换
        u_hid = HidFeatLayer(self.uNum,hid_f)(input_u);
        s_hid = HidFeatLayer(self.sNum,hid_f)(input_s);
        
        # 获取隐特征
        print(u_hid,s_hid);
        
        # baseline 
        bu = BaseLineLayer(umean)(input_u);
        bs = BaseLineLayer(smean)(input_s);
        
        
        # 连接               
        out = Concatenate()([u_hid,s_hid]);
        print(out);
        
        # 若干全连接层和dropout
        for idx,unit in enumerate(units):
            out = Dense(unit,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform,
                kernel_regularizer=l2(reg_p))(out)
            out = Dropout(drop_p)(out);
        
        out = Concatenate()([bu,out,bs]);
        print(out);
        
        out = Dense(1,
            activation=keras.activations.relu,
            kernel_initializer= glorot_uniform,
            kernel_regularizer=l2(reg_p))(out);            
        print(out);
        return Model(inputs=[input_u,input_s],outputs=out);
    
    
    
    def _get_model2(self,umean,smean):
        
        # 潜在特征
        hid_f = self.creParm.hid_feat;
        # 网络单元数
        units = self.creParm.hid_units;
        reg_p = self.creParm.reg_p;
        drop_p = self.creParm.drop_p;

        input_u = Input(shape=(1,),dtype="int32");
        input_s = Input(shape=(1,),dtype="int32");
        print(input_u);
        # one hot 转换
        u_hid = HidFeatLayer(self.uNum,hid_f)(input_u);
        s_hid = HidFeatLayer(self.sNum,hid_f)(input_s);
        
        # 获取隐特征
        print(u_hid,s_hid);
        
        # baseline 
        bu = BaseLineLayer(umean)(input_u);
        bs = BaseLineLayer(smean)(input_s);
        
        # 连接               
        out = Concatenate()([u_hid,s_hid]);
        print(out);
        
        # 若干全连接层和dropout
        for idx,unit in enumerate(units):
            out = Dense(unit,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(seed=121212+idx),
                kernel_regularizer=l2(reg_p))(out)
            out = Dropout(drop_p)(out);
        
        out = Concatenate()([bu,out,bs]);
        
        
        print(out);
        
        out = Dense(16,
            activation=keras.activations.relu,
            kernel_initializer= glorot_uniform(seed=212121),
            kernel_regularizer=l2(reg_p))(out);
        out = Dropout(drop_p)(out);
        
        
        # 输出层    
        out = Dense(1,
            activation=keras.activations.relu,
            kernel_initializer= glorot_uniform(seed=22212121),
            kernel_regularizer=l2(reg_p))(out);            
            
        print(out);
        return Model(inputs=[input_u,input_s],outputs=out);    
        
    def train(self,tp):
        train_x,train_y = reoge_data_for_keras(tp.train_data);
        test_data = reoge_data_for_keras(tp.test_data);
        
        bs = tp.batchsize;# batch_size
        lr = tp.learn_rate;# 学习律
        epoch = tp.epoch;# 迭代次数
        need_load = tp.load_cache_rec;# 是否加载记录数据
        load_path = tp.cache_rec_path;# 缓存路径
        result_path =tp.result_file_path;# 结果文件
        umean=tp.umean;
        smean=tp.smean;
        ncf_model = self._get_model(umean,smean);
        if need_load:
            ncf_model.load_weights(load_path);
        self.model = ncf_model;
        
        # 回调处理
        # 日志回调
        myhis = LossHistory(result_path);
        
        
        # early stop回调
        ear_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error',
                            min_delta=0.00002,patience=10); 
        #save回调
        chkpoint = ModelCheckpoint(load_path, monitor='val_mean_absolute_error', 
                                   save_best_only=False,
                                   save_weights_only=True,  
                                   mode='auto');
        ncf_model.compile(optimizer=Adagrad(lr), 
              loss='mae', 
              metrics=['mae']);
        
        his = ncf_model.fit(train_x,train_y,
                      batch_size=bs,epochs=epoch,
                      validation_data = test_data,
                      callbacks=[chkpoint,myhis,ear_stop],
                      verbose=0);
        val_rec = his.history['val_mean_absolute_error'];
        return min(val_rec),0;
        pass;



class simple_ncf_local():
    '''
        由keras实现的ncf_local模型
    ''' 
    # 用户服务数量
    uNum=None;
    sNum=None;
    
    # 创建参数
    creParm = None;

    # 上下文类别数
    clz_size = 0;
    
    # ncf 列表
    model = [];    

    def __init__(self,NcfCreParam):
        self.uNum,self.sNum=NcfCreParam.us_shape;
        self.creParm = NcfCreParam;
        pass;  


    def _get_model(self,umean,smean):
        
        # 潜在特征
        hid_f = self.creParm.hid_feat;
        # 网络单元数
        units = self.creParm.hid_units;
        reg_p = self.creParm.reg_p;
        drop_p = self.creParm.drop_p;

        input_u = Input(shape=(1,),dtype="int32");
        input_s = Input(shape=(1,),dtype="int32");
        print(input_u);
        # one hot 转换
        u_hid = HidFeatLayer(self.uNum,hid_f)(input_u);
        s_hid = HidFeatLayer(self.sNum,hid_f)(input_s);
        
        # 获取隐特征
        print(u_hid,s_hid);
        
        # baseline 
        bu = BaseLineLayer(umean)(input_u);
        bs = BaseLineLayer(smean)(input_s);
        
        
        # 连接               
        out = Concatenate()([u_hid,s_hid]);
        print(out);
        
        # 若干全连接层和dropout
        for idx,unit in enumerate(units):
            out = Dense(unit,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                kernel_regularizer=l2(reg_p))(out)
            out = Dropout(drop_p)(out);
        
        out = Concatenate()([bu,out,bs]);
        print(out);
        
        out = Dense(1,
            activation=keras.activations.relu,
            kernel_initializer= glorot_uniform(),
            kernel_regularizer=l2(reg_p))(out);            
        print(out);
        return Model(inputs=[input_u,input_s],outputs=out);
    
    def train(self,tp):
        
        self.clz_size = tp.classif_size;
        train_x=[];train_y=[];
        test_data=[];
        models = [];
        val_rec=[];
        
        bs = tp.batchsize;# batch_size
        lr = tp.learn_rate;# 学习律
        epoch = tp.epoch;# 迭代次数        
        need_load = tp.load_cache_rec;# 是否加载记录数据
        load_path = tp.cache_rec_path;# 缓存路径
        result_path =tp.result_file_path;# 结果文件
        umean=tp.umean;
        smean=tp.smean;
    
    
        # 创建过程
        for clz in range(self.clz_size):
            
            t_x,t_y = reoge_data_for_keras(tp.train_data[clz]);
            t_data = reoge_data_for_keras(tp.test_data[clz]);
            train_x.append(t_x);
            train_y.append(t_y);
            test_data.append(t_data);
            
            ncf_model = self._get_model(umean,smean);

            ncf_model.compile(optimizer=Adagrad(lr[clz]), 
                  loss='mae', 
                  metrics=['mae']);
            models.append(ncf_model);
            
            
        # 回调处理
        # 日志回调
        myhis = LossHistory(result_path);
        
        for ep in range(epoch):
            sum_up=0;
            sum_dw=0;
            fwrite_append(result_path, '');# 置空行
            for clz in range(self.clz_size):
                print('--->ep=%d \tclz=%d'%(ep,clz));
                ds =  len(test_data[clz][1]);
                his = ncf_model.fit(train_x[clz],train_y[clz],
                              batch_size=bs,epochs=1,
                              validation_data = test_data[clz],
                              callbacks=[myhis],
                              verbose=0);
                mae_rec = his.history['val_mean_absolute_error'][-1];
                sum_up +=  mae_rec* ds;
                sum_dw += ds;
            ep_mae = sum_up/sum_dw;
            val_rec.append(ep_mae);
            ep_str = 'ep=%d \tmae=%.6f\n'%(ep,ep_mae);
            
            fwrite_append(result_path, ep_str);# 置空行
            print(ep_str);
        
        return min(val_rec),0;
        pass;
   


if __name__ == '__main__':
    
#     c = NcfCreParam();
#     c.us_shape=(339,5825);
#     c.hid_feat=32;# 隐含矩阵特征数
#     c.hid_units=[64,32,16]; # 网络各个隐层隐含单元数
#     c.reg_p = 0.01;
#     c.drop_p=0.001;
#     
#     m = simple_ncf(c);
    
    
    pass