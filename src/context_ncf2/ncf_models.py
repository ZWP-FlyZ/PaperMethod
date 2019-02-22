# -*- coding: utf-8 -*-
'''
Created on 2019年2月16日

@author: zwp12
'''

import tensorflow as tf;
import time;
from tools.ncf_tools import reoge_data_for_context,reoge_data_for_context_
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
        


class HidFeatLayer(Layer):
    def __init__(self, in_dim,out_dim,**kwargs):
        self.output_dim = out_dim;
        self.in_dim = in_dim;
        super(HidFeatLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
               
        self.ker = self.add_weight(name='hid_feat', 
                                      shape=(self.in_dim, self.output_dim),
                                      initializer=glorot_uniform(),
                                      dtype='float32',
                                      trainable=True);                                      
        super(HidFeatLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        print(x);
        choosed = K.gather(self.ker, K.cast(K.squeeze(x, 1),'int32'));
        print(choosed);
        return choosed;

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class BiasLayer(Layer):
    def __init__(self, input_dim,**kwargs):
        self.output_dim = 1;
        self.input_dim = input_dim;
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
               
        self.b = self.add_weight(name='B', 
                                      shape=(self.input_dim, self.output_dim),
                                      initializer=zero,
                                      dtype='float32',
                                      trainable=True);                                      
        super(BiasLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, u):
        choosed = K.gather(self.b, K.cast(K.squeeze(u, 1),'int32'));
        print(choosed);
        return choosed;

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class context_ncf():
    '''
    由keras实现的ncf模型,输入包含u,s 和uw,sw
    '''
    
    # 用户服务数量
    uNum=None;
    sNum=None;
    uCluNum=None;# 用户簇数量
    sCluNum=None;# 服务簇数量
    
    # 创建参数
    creParm = None;

    # ncf 模型
    model = None;
    
    def __init__(self,NcfCreParam):
        self.uNum,self.sNum=NcfCreParam.us_shape;
        self.uCluNum,self.sCluNum=NcfCreParam.clu_num;
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

        # U,S 输入
        input_u = Input(shape=(1,),dtype="int32");
        input_s = Input(shape=(1,),dtype="int32");
        
        # U,S 对应的隶属度输入
        input_uw = Input(shape=(self.uCluNum,),dtype="float32");
        input_sw = Input(shape=(self.sCluNum,),dtype="float32");
        
        print(input_uw);
        
        # U,S交互的潜在特征
        u_hid = HidFeatLayer(self.uNum,hid_f)(input_u);
        s_hid = HidFeatLayer(self.sNum,hid_f)(input_s);
        
        # 隶属度潜在特征
        uw_hid = Dense(hid_f,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                    kernel_regularizer=l2(reg_p))(input_uw);
        sw_hid = Dense(hid_f,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                    kernel_regularizer=l2(reg_p))(input_sw);
        

        
                    
        print(sw_hid);
        # 连接               
        out = Concatenate()([u_hid,s_hid,uw_hid,sw_hid]);
#         out = Concatenate()([u_hid,s_hid]);
        print(out);
        
        # 若干全连接层和dropout
        for _,unit in enumerate(units):
            out = Dense(unit,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                kernel_regularizer=l2(reg_p))(out)
            out = Dropout(drop_p)(out);
        
#         out = Concatenate()([out,uw_hid,sw_hid]);
        print(out);
        
        out = Dense(1,
            activation=keras.activations.relu,
            kernel_initializer= glorot_uniform())(out);            
        print(out);
        return Model(inputs=[input_u,input_s,input_uw,input_sw],outputs=out);
        
    def train(self,tp):
        train_x,train_y = reoge_data_for_context(tp.train_data,tp.fcm_ws);
        test_data = reoge_data_for_context(tp.test_data,tp.fcm_ws);
        
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
                      verbose=1);
        val_rec = his.history['val_mean_absolute_error'];
        return min(val_rec),0;
        pass;

class context_ncf_bais():
    '''
    由keras实现的ncf模型,输入包含u,s 和uw,sw
    带偏置
    '''
    
    # 用户服务数量
    uNum=None;
    sNum=None;
    uCluNum=None;# 用户簇数量
    sCluNum=None;# 服务簇数量
    
    # 创建参数
    creParm = None;

    # ncf 模型
    model = None;
    
    def __init__(self,NcfCreParam):
        self.uNum,self.sNum=NcfCreParam.us_shape;
        self.uCluNum,self.sCluNum=NcfCreParam.clu_num;
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

        
        # U,S 输入
        input_u = Input(shape=(1,),dtype="int32");
        input_s = Input(shape=(1,),dtype="int32");
        
        # U,S 对应的隶属度输入
        input_uw = Input(shape=(self.uCluNum,),dtype="float32");
        input_sw = Input(shape=(self.sCluNum,),dtype="float32");
        
        print(input_uw);
        
        # U,S交互的潜在特征
        u_hid = HidFeatLayer(self.uNum,hid_f)(input_u);
        s_hid = HidFeatLayer(self.sNum,hid_f)(input_s);
        
        # 隶属度潜在特征
        uw_hid = Dense(hid_f,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                    kernel_regularizer=l2(reg_p))(input_uw);
        sw_hid = Dense(hid_f,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                    kernel_regularizer=l2(reg_p))(input_sw);
        
        
        # 偏置层
        bu = BiasLayer(self.uNum)(input_u);
        bs = BiasLayer(self.sNum)(input_s);            
        print(sw_hid);
        # 连接               
        out = Concatenate()([u_hid,s_hid,uw_hid,sw_hid]);
#         out = Concatenate()([u_hid,s_hid]);
        print(out);
        
        # 若干全连接层和dropout
        for _,unit in enumerate(units):
            out = Dense(unit,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                kernel_regularizer=l2(reg_p))(out)
            out = Dropout(drop_p)(out);
        
#         out = Concatenate()([out,uw_hid,sw_hid]);
        print(out);
        out = Concatenate()([bu,out,bs]);
        
        out = Dense(1,
            activation=keras.activations.relu,
            kernel_initializer= glorot_uniform())(out);            
        print(out);
        return Model(inputs=[input_u,input_s,input_uw,input_sw],outputs=out);
        
    def train(self,tp):
        train_x,train_y = reoge_data_for_context(tp.train_data,tp.fcm_ws);
        test_data = reoge_data_for_context(tp.test_data,tp.fcm_ws);
        
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
                      verbose=1);
        val_rec = his.history['val_mean_absolute_error'];
        return min(val_rec),0;
        pass;

class context_ncf2():
    '''
    由keras实现的ncf模型,输入包含u,s 和uw,sw
    交叉模型
    '''
    
    # 用户服务数量
    uNum=None;
    sNum=None;
    uCluNum=None;# 用户簇数量
    sCluNum=None;# 服务簇数量
    
    # 创建参数
    creParm = None;

    # ncf 模型
    model = None;
    
    def __init__(self,NcfCreParam):
        self.uNum,self.sNum=NcfCreParam.us_shape;
        self.uCluNum,self.sCluNum=NcfCreParam.clu_num;
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

        # U,S 输入
        input_u = Input(shape=(1,),dtype="int32");
        input_s = Input(shape=(1,),dtype="int32");
        
        # U,S 对应的隶属度输入
        input_uw = Input(shape=(self.uCluNum,),dtype="float32");
        input_sw = Input(shape=(self.sCluNum,),dtype="float32");
        
        print(input_uw);
        
        # U,S交互的潜在特征
        u_hid = HidFeatLayer(self.uNum,hid_f)(input_u);
        s_hid = HidFeatLayer(self.sNum,hid_f)(input_s);
        
        # 隶属度潜在特征
        uw_hid = Dense(hid_f,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                    kernel_regularizer=l2(reg_p))(input_uw);
        sw_hid = Dense(hid_f,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                    kernel_regularizer=l2(reg_p))(input_sw);
            
        print(sw_hid);
        # 连接               
        out = Concatenate()([u_hid,s_hid]);
        out2 = Concatenate()([uw_hid,sw_hid]);
#         out = Concatenate()([u_hid,s_hid]);
        print(out);
        
        out = Dense(16,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                kernel_regularizer=l2(reg_p))(out)
                
        out2 = Dense(16,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                kernel_regularizer=l2(reg_p))(out2)        

        
        out = Concatenate()([out,out2]);
        
        out = Dense(16,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                kernel_regularizer=l2(reg_p))(out)
        
        print(out);
        
        out = Dense(1,
            activation=keras.activations.relu,
            kernel_initializer= glorot_uniform())(out);            
        print(out);
        return Model(inputs=[input_u,input_s,input_uw,input_sw],outputs=out);
        
    def train(self,tp):
        train_x,train_y = reoge_data_for_context(tp.train_data,tp.fcm_ws);
        test_data = reoge_data_for_context(tp.test_data,tp.fcm_ws);
        
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
                            min_delta=0.0002,patience=4); 
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
                      verbose=1);
        val_rec = his.history['val_mean_absolute_error'];
        return min(val_rec),0;
        pass;


class context_ncf_():
    '''
    由keras实现的ncf模型,输入只包含上下文信息
    '''
    
    # 用户服务数量
    uNum=None;
    sNum=None;
    uCluNum=None;# 用户簇数量
    sCluNum=None;# 服务簇数量
    
    # 创建参数
    creParm = None;

    # ncf 模型
    model = None;
    
    def __init__(self,NcfCreParam):
        self.uNum,self.sNum=NcfCreParam.us_shape;
        self.uCluNum,self.sCluNum=NcfCreParam.clu_num;
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

        

        
        input_uw = Input(shape=(self.uCluNum,),dtype="float32");
        input_sw = Input(shape=(self.sCluNum,),dtype="float32");
        
        print(input_uw);
        # one hot 转换
        
        
        
        uw_hid = Dense(hid_f,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                    kernel_regularizer=l2(reg_p))(input_uw);
        sw_hid = Dense(hid_f,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                    kernel_regularizer=l2(reg_p))(input_sw);
                    
        print(sw_hid);
        # 连接               
        out = Concatenate()([uw_hid,sw_hid]);
        print(out);
        
        # 若干全连接层和dropout
        for _,unit in enumerate(units):
            out = Dense(unit,
                activation=keras.activations.relu,
                kernel_initializer= glorot_uniform(),
                kernel_regularizer=l2(reg_p))(out)
            out = Dropout(drop_p)(out);
        
        print(out);
        
        out = Dense(1,
            activation=keras.activations.relu,
            kernel_initializer= glorot_uniform())(out);            
        print(out);
        return Model(inputs=[input_uw,input_sw],outputs=out);
        
    def train(self,tp):
        train_x,train_y = reoge_data_for_context_(tp.train_data,tp.fcm_ws);
        test_data = reoge_data_for_context_(tp.test_data,tp.fcm_ws);
        
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
                      verbose=1);
        val_rec = his.history['val_mean_absolute_error'];
        return min(val_rec),0;
        pass;


if __name__ == '__main__':
    pass