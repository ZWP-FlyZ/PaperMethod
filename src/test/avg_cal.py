# -*- coding: utf-8 -*-
'''
Created on 2018年12月17日

@author: zwp
'''
import numpy as np;


def run():
    edge=5825;
    pn=8;
    up = np.zeros(8);
    sp_area = edge*edge/2/pn;
    x_res = [];
    for i in range(pn-1):
        b = 2*up[i];
        c = -2*sp_area;
        x = (-b + np.sqrt(b*b - 4*c))/2.0;
        x_res.append(x);
        up[i+1]=x+up[i];

    x_res.append(5825-up[-1]);
    print(up);
    print(np.sort(np.array(x_res,int)));    
    pass;


if __name__ == '__main__':
    run();
    pass