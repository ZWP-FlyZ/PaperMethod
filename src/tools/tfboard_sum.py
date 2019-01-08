# -*- coding: utf-8 -*-
'''
Created on 2018年8月9日

@author: zwp
'''

import tensorflow as tf;

def var_summaries(var):
    with tf.name_scope('summaries'):
        mean  = tf.reduce_mean(var);
        tf.summary.scalar('mean',mean);
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)));
        tf.summary.scalar('stddev',stddev);
        tf.summary.histogram('hist',var);


if __name__ == '__main__':
    pass