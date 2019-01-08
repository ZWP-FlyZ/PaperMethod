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


from tensorflow.python.keras import backend as K;
from tensorflow.python.keras.layers import Input,Lambda,Dense,Concatenate,Dropout
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras.regularizers import l2;
from tensorflow.python.keras.optimizers import Adagrad;
from tensorflow.python.keras import Model;
from tensorflow.python.keras.callbacks import ModelCheckpoint

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
    
    def _get_model(self):
        
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
        U = Lambda(lambda x:K.one_hot(K.cast(K.squeeze(x,1),"int32"),self.uNum))(input_u);
        S = Lambda(lambda x:K.one_hot(K.cast(K.squeeze(x,1),"int32"),self.sNum))(input_s);
        print(U,S);
        
        
        # 获取隐特征
        u_hid = Dense(hid_f,use_bias=False,
                       kernel_initializer = glorot_uniform(seed=123456))(U);
        s_hid = Dense(hid_f,use_bias=False,
                       kernel_initializer = glorot_uniform(seed=654321))(S);
        print(u_hid,s_hid);
        # 连接               
        out = Concatenate()([u_hid,s_hid]);
        print(out);
        
        # 若干全连接层和dropout
        for idx,unit in enumerate(units):
            out = Dense(unit,
                activation=tf.keras.activations.relu,
                kernel_initializer= glorot_uniform(seed=121212+idx),
                kernel_regularizer=l2(reg_p))(out)
            out = Dropout(drop_p)(out);
        
        print(out);
        
        out = Dense(1,
            activation=tf.keras.activations.relu,
            kernel_initializer= glorot_uniform(seed=212121),
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
        
        # early stop回调
        #save回调
        chkpoint = ModelCheckpoint('./chk.h5', monitor='val_loss', 
                                   save_best_only=False,  
                                   mode='auto');
        
        ncf_model.compile(optimizer=Adagrad(lr), 
              loss='mae', 
              metrics=['mae']);
        
        ncf_model.fit(train_x,train_y,
                      batch_size=bs,epochs=epoch,
                      validation_data = test_data,
                      callbacks=[chkpoint]);
        
        pass;
                           
    
        

class ncf_pp_local():
    '''
    >混合模型 地理位置信息
    '''
    # 用户服务输入
    feature=None;# tf.placeholer=[[u,s],[u,s]......];
    # 响应时间 
    label=None; # [[rt],[rt]...]
    
    # 创建参数
    create_param = None;
    
    # 预测结果
    Py = None;
    # 损失
    loss = None;
    
    # 评测结果
    mae = None;
    rmse= None;
    
    # 数据迭代器
    data_iter=None;
    
    Yj=None;
    
    def __init__(self,NcfCreParam):
        self.uNum,self.sNum=NcfCreParam.us_shape;
        self.create_param = NcfCreParam;
        pass;
    
    
    def toOneHot(self,feature):
        U = tf.one_hot(feature[:,0], self.uNum)
        S = tf.one_hot(feature[:,1],self.sNum);
        return U,S;
    
    def create_model(self,feat,Y,NcfCreParam,us_invked):
        hid_f = NcfCreParam.hid_feat;                          
        hid_units = NcfCreParam.hid_units;
        
        hid_actfunc =tf.nn.relu;
        # out_actfunc = tf.sigmoid;

        reg_func = tf.contrib.layers.l2_regularizer(NcfCreParam.reg_p);
        
        
        # ################# 处理 u_invked ###################### #
        # yj独立
        Yj = tf.Variable(tf.zeros(shape=[self.sNum,hid_f]),name='Yj');
        
        us_invked = tf.constant(us_invked,tf.float32);
        
        idx = tf.reshape(tf.cast(feat[:,0],tf.int32),[-1,1]);
        
        uivk = tf.gather_nd(us_invked,idx);
        
        Zj = tf.matmul(uivk,Yj);
        
        
        U,S = self.toOneHot(feat);
        
        # 初始化 隐含特征矩阵
        Pu = tf.layers.dense(inputs=U,units=hid_f,
                           #  kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        Qs = tf.layers.dense(inputs=S,units=hid_f,
                             # kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);

        bu = tf.layers.dense(inputs=U,units=1,
                             kernel_initializer=tf.initializers.zeros,
                            kernel_regularizer=reg_func);
        
        bi = tf.layers.dense(inputs=S,units=1,
                            kernel_initializer=tf.initializers.zeros,
                            kernel_regularizer=reg_func);

        mu = tf.fill(tf.shape(bu),tf.cast(NcfCreParam.baseline_mu,tf.float32));
        # mu = tf.constant(NcfCreParam.baseline_mu,dtype=tf.float32,shape=tf.shape(bu));
        # baseline=tf.reshape(baseline,[-1,1]);
        Zu = Pu+Zj;

        out = tf.concat([Zu,Qs],axis=1); 
     
        
        for unit in hid_units:
            out=tf.layers.dense(inputs=out,units=unit,
                                activation=hid_actfunc,
                                kernel_regularizer=reg_func);
            out=tf.layers.dropout(out,NcfCreParam.drop_p);                    
        
        # 双模型混合                        
        out =  tf.concat([out,bu,bi,mu],axis=1);
        

        # 输出层                       
        out=tf.layers.dense(inputs=out,units=1,
                            activation=hid_actfunc,
                            kernel_regularizer=reg_func);
        print(out)
        Py=out;                    
        # 误差                   
        # loss = tf.reduce_mean(tf.losses.huber_loss(Y,out));
        loss = tf.reduce_sum(tf.abs(Y-out));
        # tf.summary.scalar('loss',loss);
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        # tf.summary.scalar('mae',mae);
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        # tf.summary.scalar('rmse',rmse);
        return Py,loss,mae,rmse; 
    ############################# end  ##################################    
    
    
    def train(self,NcfTraParm):

        k = NcfTraParm.classif_size;
        class_parm = [{} for _ in range(k)];
        
        for i in range(k):
            with tf.name_scope('class%d'%(i)):
                
                train_data = reoge_data(NcfTraParm.train_data[i]);
                test_data = reoge_data(NcfTraParm.test_data[i]);
                testn = len(test_data[0]);
                global_step = tf.Variable(0,trainable=False,name='gs');
                class_parm[i]['global_step'] = global_step;
                ds = tf.data. \
                        Dataset.from_tensor_slices(train_data);
                ds = ds.shuffle(1000).batch(NcfTraParm.batch_size);
                
                test_ds = tf.data.Dataset.from_tensor_slices(test_data);
                test_ds = test_ds.batch(testn);
                it = tf.data.Iterator.from_structure(ds.output_types,
                                                    ds.output_shapes);
                
                feat,Y = it.get_next(); 
                train_init_op = it.make_initializer(ds);
                test_init_op = it.make_initializer(test_ds);   
                
                class_parm[i]['train_init_op'] = train_init_op;
                class_parm[i]['test_init_op'] = test_init_op;
                    
                _,loss,tmae,trmse= self.create_model(feat, Y, self.create_param,NcfTraParm.us_invked[i]);
                
                class_parm[i]['loss'] = loss;
                class_parm[i]['tmae'] = tmae;
                class_parm[i]['trmse'] = trmse;
                
                # loss+=tf.losses.get_regularization_loss();
                
                lr = tf.train.exponential_decay(NcfTraParm.learn_rate, global_step,
                                        NcfTraParm.lr_decy_step,
                                        NcfTraParm.lr_decy_rate,
                                        staircase=True);
                                        
                train_step = tf.train.AdagradOptimizer(lr). \
                            minimize(loss, global_step );
                
                class_parm[i]['train_step'] = train_step;
                
        # summ_meg = tf.summary.merge_all();
        save = tf.train.Saver();
        with tf.Session() as sess:
            # train_summ = tf.summary.FileWriter(NcfTraParm.summary_path+'/train',sess.graph);
            # test_summ =tf.summary.FileWriter(NcfTraParm.summary_path+'/test');
            if NcfTraParm.load_cache_rec:
                save.restore(sess,NcfTraParm.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
            
            now = time.time();
            eptime = now;
            for ep in range(NcfTraParm.epoch):
                test_tmae=0.0;cot=0;
                for i in range(k):
                    sess.run(class_parm[i]['train_init_op']);
                    while True:
                        try:
                            _,vloss,gs=sess.run((class_parm[i]['train_step'],
                                                 class_parm[i]['loss'],
                                                 class_parm[i]['global_step']));
                            if gs%(500) == 0:
                                print('ep%d\t class%d\t loopstep:%d\t time:%.2f\t loss:%f'%(ep,i,gs,time.time()-now,vloss))
                                # summ = sess.run((summ_meg));
                                # train_summ.add_summary(summ, gs);
                                now=time.time();
                        except tf.errors.OutOfRangeError:
                            break  
                    sess.run(class_parm[i]['test_init_op']);
                    # summ,vmae,vrmse,vloss=sess.run((summ_meg,tmae,trmse,loss));
                    vmae,vrmse,vloss=sess.run((class_parm[i]['tmae'],
                                               class_parm[i]['trmse'],
                                               class_parm[i]['loss']));
                    test_tmae+=vmae*len(NcfTraParm.test_data[i]);
                    cot+=len(NcfTraParm.test_data[i]);
                    print(vmae);
                    # test_summ.add_summary(summ, ep);
                vmae=test_tmae/cot;
                eps = '==================================================\n'
                eps += 'ep%d结束 \t eptime=%.2f\n' %(ep ,time.time()-eptime);
                eps += 'test_mae=%f test_rmse=%f\n'%(vmae,vrmse);
                eps += 'acttime=%s\n'%(time.asctime());
                eps += '==================================================\n'
                eptime = time.time();
                print(eps);
                if NcfTraParm.result_file_path != '': 
                    fwrite_append(NcfTraParm.result_file_path,eps);                
                 
                print('ep%d结束 \t eponloss=%f\t test_mae=%f test_rmse=%f\n'%(ep ,vloss,vmae,vrmse));
            
            if NcfTraParm.cache_rec_path != '':
                save.save(sess,NcfTraParm.cache_rec_path)
        pass;
    
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