# -*- coding: utf-8 -*-
"""
run file for neural walker

@author: hongyuan
"""
from memory_profiler import profile
import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
import modules.utils as utils
import modules.models as models
import modules.optimizers as optimizers
import modules.trainers as trainers
#import modules.data_processers as data_processers
import data_processers as data_processers

import run_model
import datetime
import argparse
__author__ = 'Hongyuan Mei'

dtype=theano.config.floatX



#@profile
def main():

    parser = argparse.ArgumentParser()
    #
    '''
    modify here accordingly ...
    '''
    #
    parser.add_argument('--FileData', required=False, help='Path of the dataset')
    #
    parser.add_argument('--DimModel', required=False, help='Dimension of LSTM model ')
    parser.add_argument('--Seed', required=False, help='Seed of random state')
    parser.add_argument('--Optimizer', required=False, help='Optimizer of training')
    parser.add_argument('--DropOut', required=False, help='Drop-out rate')

    parser.add_argument('--cur_epoch', type=int, default=0, help='Training epoch')
    parser.add_argument('--start_idx', type=int, default=0, help='Start step')
    parser.add_argument('--train_steps', type=int, default=100, help='Train steps of an epoch')
    parser.add_argument('--data_fold', type=int, default=-1, help='Data fold')
    parser.add_argument('--databag', default='sail', help='Name of dataset')
    parser.add_argument('--train_all', type=int, default=0, help='Train all sents one epoch by another')
    #
    args = parser.parse_args()
    #
    if args.FileData == None:
        args.FileData = None

    if args.Optimizer == None:
        args.Optimizer = 'adam'
    else:
        args.Optimizer = args.Optimizer
    #
    if args.DimModel == None:
        args.DimModel = numpy.int32(100)
    else:
        args.DimModel = numpy.int32(args.DimModel)
    if args.Seed == None:
        args.Seed = numpy.int32(12345)
    else:
        args.Seed = numpy.int32(args.Seed)
    if args.DropOut == None:
        args.DropOut = numpy.float32(0.9)
    else:
        args.DropOut = numpy.float32(args.DropOut)
    #
    if args.Map1 == None:
        args.Map1 = 'grid'
    else:
        args.Map1 = str(args.Map1)
    if args.Map2 == None:
        args.Map2 = 'jelly'
    else:
        args.Map2 = str(args.Map2)
    assert args.start_idx >= 0

    input_trainer = {
        'random_seed': args.Seed,
        'path_rawdata': args.FileData,
        'drop_out_rate': args.DropOut,
        'dim_model': args.DimModel,
        'optimizer': args.Optimizer,
        'save_file':'./results/ts_results/%s/model.pkl'%args.databag,
        'result_dir':'./results/ts_results/%s/result.txt'%args.databag
    }
    if args.train_all:
        input_trainer['save_file'] = './results/%s/fold%d_model.pkl'%(args.databag, args.data_fold)
        input_trainer['result_dir'] = './results/%s/fold%d_ep%d.txt'%(args.databag, args.data_fold, args.cur_epoch)
    print '\nTraining epoch: %d    start index: %d\n'%(args.cur_epoch, args.start_idx)
    run_model.train_model(input_trainer, args.cur_epoch, args.start_idx, args.train_steps, 
        args.data_fold, args.databag)


if __name__ == "__main__": 
    #import ipdb
    #ipdb.set_trace()
    main()
