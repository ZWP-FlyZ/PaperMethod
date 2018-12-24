# -*- coding: utf-8 -*-
'''
Created on 2018年2月3日

@author: zwp
'''

# 加载用户地理位置信息
def loadLocation(path):
    locs={};
    with open(path) as f:
        for line in f:
            userId, _, loc = line.strip().split('\t');
            uid = int(userId)-1;
            locs.setdefault(uid, {});
            locs[uid]=loc;
    return  locs;   
    




