# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: zwp12
'''


def arr2str(arr):
    res = '';
    if arr is None or len(arr)==0:
        raise ValueError('err arr');
    res += str(arr[0]);
    for i in range(1,len(arr)):
        res+=','+str(arr[i]);
    return res;

def str2arr(str):
    res = [];
    if str is None or str=='':
        raise ValueError('err str');
    vs = str.split(',');
    for i in vs:
        res.append(int(i));
    return res;
