# -*- coding: utf-8 -*-
'''
Created on 2018年8月9日

@author: zwp12
'''
import os;

def fwrite_append(file_path,strmsg):
    '''
    >如果过存在文件，则在文件末尾添加消息
    >如果不存在则创建一个
    '''
    pn,_ = os.path.split(file_path);
    if not os.path.isdir(pn):
        os.makedirs(pn)
    with open(file_path,'a+') as f:
        f.write(str(strmsg)+'\n');

def fwrite(file_path,strmsg):
    '''
    >如果存在文件，则清空内容后写
    >如果不存在则创建一个
    '''
    pn,_ = os.path.split(file_path);
    if not os.path.isdir(pn):
        os.makedirs(pn)    
    with open(file_path,'w') as f:
        f.write(str(strmsg)+'\n');


if __name__ == '__main__':
    

    
    pass