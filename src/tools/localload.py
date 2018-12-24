# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: zwp12
'''


def load(path):
    locs={};
    with open(path) as f:
        for line in f:
            ids, AS, loc = line.strip().split('\t');
            uid = int(ids)-1;
            locs[uid]=(AS,loc);
    return  locs;       

def load_location_name(path):
    locs=set();
    with open(path) as f:
        for line in f:
            _, _, loc = line.strip().split('\t');
            locs.add(loc);
    return  locs; 

def load_locmore(path):
    res={};
    with open(path) as f:
        for line in f:
            loc, fag,ast = line.strip().split('\t');
            lat,lgt = ast.split(', ');
            lat=float(lat);
            lgt=float(lgt);
            res[loc]=(fag,[lat,lgt]);
    return res;

