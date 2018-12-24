# -*- coding: utf-8 -*-
'''
Created on 2018年2月2日

@author: zwp
'''

import platform;


def check():
    sysstr = platform.system();
    if(sysstr =="Windows"):
        return 'w';
    elif(sysstr == "Linux"):
        return 'l';
    else:
        return 'o';





