# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: zwp12
'''


def load(path):
    locs={};
    with open(path) as f:
        for line in f:
            ids, AS, loc,_,_,ip = line.strip().split('\t');
            ip=ip.split('.');
            ip = list(map(lambda x:int(x),ip));            
            uid = int(ids);
            locs[uid]=(AS,loc,ip);
    return  locs;       

def load_location_name(path):
    locs=set();
    with open(path) as f:
        for line in f:
            _, _, loc,_,_,ip = line.strip().split('\t');
            locs.add(loc);
    return  list(locs); 

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

def load_userinfo(path):
    locs={};
    with open(path) as f:
        for line in f:
            ids, AS, loc,lat,lngt,ip = line.strip().split('\t');
            ip=ip.split('.');
            ip = list(map(lambda x:int(x),ip));
            uid = int(ids);
            locs[uid]=(AS,loc,[float(lat),float(lngt)],ip);
    return  locs;


