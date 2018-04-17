# -*- coding: utf-8 -*-
"""
run the neural walker model

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
import modules.beam_search as beam_search


dtype=theano.config.floatX

#TODO: function to train seq2seq models
#@profile
def train_model(input_trainer, cur_epoch, start_idx, train_steps, data_fold, databag):
    '''
    this function is called to train model
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(
        input_trainer['random_seed']#12345
    )

    #TODO: get the data and process the data
    print "reading and processing data ... "
    print "databag: %s  data_fold: %d\n"%(databag, data_fold)

    data_process = data_processers.DataProcess(
        data_fold = data_fold,
        databag = databag,
        path_rawdata = input_trainer['path_rawdata']
    )
    
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    model_settings = {
        'validate': True,
        'load_model': True, #False, #
        'save_file': input_trainer['save_file'],
        'dim_lang': data_process.dim_lang,
        #'dim_world': data_process.dim_world,
        'dim_action': data_process.dim_action,
        'dim_model': input_trainer['dim_model'],
        'optimizer': input_trainer['optimizer'],
        'drop_out_rate': input_trainer['drop_out_rate']
    }
    if cur_epoch == 0 and start_idx == 0:
        model_settings['load_model'] = False

    trainer = trainers.NeuralWalkerTrainer(model_settings = model_settings)

    compile_end = time.time()
    compile_time = compile_end - compile_start
    print "model finished, comilation time is ", round(compile_time, 0)


    # start training
    train_start = time.time()
    err = 0.0
    train_counter = 0
    #train_steps = 100 
    start_idx *= train_steps
    end_idx = start_idx + train_steps
    print '\n\nend_idx = %d\n\n'%end_idx
    if end_idx > len(data_process.dict_data['instruction']):
        end_idx = len(data_process.dict_data['instruction'])
    for idx_data in xrange(start_idx, end_idx):
        #ipdb.set_trace()
        if idx_data not in data_process.devset:
            data_process.process_one_data(idx_data)
            cost_numpy = trainer.model_learn(
                data_process.seq_lang_numpy,
                data_process.seq_action_numpy)
            err += cost_numpy
            train_counter += 1
            #if idx_data % 100 == 99:
            print 'training %d-th instruction'%idx_data
    train_err = err / train_counter
    train_end = time.time()

    # 
    trainer.save_model()  
    print "finish training"
    if model_settings['validate']:
        validate_model(
            input_trainer['result_dir'], end_idx, data_process, trainer)
    # function finished
# training finished


def validate_model(result_dir, end_idx, data_process, trainer):
    print "validating ... "
    ##
    #TODO: build another data process for Greedy search, i.e., gs
    ##
    bs_settings = {
        'size_beam': 1, # greedy search
        'path_model': None,
        'trained_model': trainer.get_model(), 
        'dim_lang': data_process.dim_lang,
        'map': None
    }
    #
    bs = beam_search.BeamSearchNeuralWalker(bs_settings)
    
    dev_start = time.time()
    dev_steps = 100 
    total_steps = len(data_process.dict_data['instruction'])
    end_dev_idx = end_idx + dev_steps
    if end_dev_idx > total_steps:
        end_dev_idx = total_steps

    right_act = 0
    wrong_act = 0
    total_act = 0
    tagged_act = 0
    action_sequence = []
    import ipdb
    #ipdb.set_trace()
    f = open(result_dir, 'a')

    dev_counter = 0
    for idx_data in data_process.devset: #xrange(end_idx, end_dev_idx):
        data_process.process_one_data(idx_data)
        bs.set_encoder(data_process.seq_lang_numpy, data_process.seq_action_numpy)
        bs.init_beam() # will append ht and ct once
        actions = bs.search_func()
        #print 'Sentence %d: %s\n'%(idx_data, ' '.join(data_process.dict_data['instruction'][idx_data]))
        #print 'Right action: %s\n'%' '.join(data_process.action_word)
        #print 'Tagged action: %s\n\n'%' '.join([data_process.ind2word[int(ai)] for ai in actions])
        
        #f.write('Sentence %d: %s\n'%(idx_data, ' '.join(data_process.dict_data['instruction'][idx_data])))
        #f.write('Right action: %s\n'%' '.join(data_process.action_word))
        #f.write('Tagged action: %s\n\n'%' '.join([data_process.ind2word[int(ai)] for ai in actions]))
        action_sequence.append(actions)
        tagged_act += len(actions)
        total_act += len(data_process.seq_action_numpy)
        for ai in actions:
            if data_process.ind2word[ai] in data_process.action_word:
                right_act += 1
            else:
                wrong_act += 1
        dev_counter += 1
        #if counter >= dev_steps:
        #    break
    #ipdb.set_trace()
    precision = 0
    f_measure = 0
    recall = 0
    if total_act > 0:
        if tagged_act > 0:
            precision = float(right_act)/tagged_act
            recall = float(right_act)/total_act
        if precision+recall > 0:
            f_measure = 2*precision*recall/(precision+recall)
    else:
        print '\n\n----- total_act is none! -----\n\n'
        f.write('\n\n----- total_act is none! -----\n\n')
    f.write('\nend_idx : %d\n'%end_idx)
    f.write('total_action: %d\nright_action: %d\ntag_action: %d\n'%(total_act, right_act, tagged_act))
    f.write('recall: %f\nprecision: %f\nf_measure: %f\n\n'%(recall, precision, f_measure))
    print 'recall: %f\nprecision: %f\nf_measure: %f'%(recall, precision, f_measure)
    dev_end = time.time()



def test_model(input_tester):
    '''
    this function is called to test model
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(12345)
    #

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcess(
        path_rawdata=input_tester['path_rawdata']
    )
    #
    #TODO: build another data process for Greedy search, i.e., gs
    ##
    bs_settings = {
        'size_beam': 1, # greedy search
        'path_model': input_tester['path_model'],
        'trained_model': None,
        'dim_lang': data_process.dim_lang,
        'map': data_process.maps[
            data_process.map2idx[input_tester['map_test']]
        ]
    }
    #
    #TODO: build the model
    print "building model ... "
    #
    bs = beam_search.BeamSearchNeuralWalker(
        bs_settings
    )
    #
    name_map = input_tester['map_test']
    #
    cnt_success = 0
    num_steps = len(
        data_process.dict_data['dev'][name_map]
    ) + len(
        data_process.dict_data['train'][name_map]
    )
    #
    bs = beam_search.BeamSearchNeuralWalker(
        bs_settings
    )
    #
    bs_results = []
    #
    for idx_data, data in enumerate(data_process.dict_data['dev'][name_map]):
        data_process.process_one_data(
            idx_data, name_map, 'dev'
        )
        #import pdb
        #pdb.set_trace()
        bs.set_encoder(
            data_process.seq_lang_numpy,
            data_process.seq_world_numpy
        )
        pos_start, pos_end = data_process.get_pos(
            idx_data, name_map, 'dev'
        )
        bs.init_beam(
            numpy.copy(pos_start), numpy.copy(pos_end)
        )
        bs.search_func()
        #
        if bs.check_pos_end():
            cnt_success += 1
        #
        result = {
            'path_ref': data['cleanpath'],
            'path_gen': bs.get_path(),
            'success': bs.check_pos_end(),
            'pos_current': bs.finish_list[0]['pos_current'],
            'pos_destination': bs.finish_list[0]['pos_destination']
        }
        bs_results.append(result)
        #
        bs.refresh_state()
        #
    #
    #
    for idx_data, data in enumerate(data_process.dict_data['train'][name_map]):
        data_process.process_one_data(
            idx_data, name_map, 'train'
        )
        bs.set_encoder(
            data_process.seq_lang_numpy,
            data_process.seq_world_numpy
        )
        pos_start, pos_end = data_process.get_pos(
            idx_data, name_map, 'train'
        )
        bs.init_beam(
            numpy.copy(pos_start), numpy.copy(pos_end)
        )
        bs.search_func()
        #
        if bs.check_pos_end():
            cnt_success += 1
        #
        result = {
            'path_ref': data['cleanpath'],
            'path_gen': bs.get_path(),
            'success': bs.check_pos_end(),
            'pos_current': bs.finish_list[0]['pos_current'],
            'pos_destination': bs.finish_list[0]['pos_destination']
        }
        bs_results.append(result)
        #
        #
        bs.refresh_state()
        ##
    #
    #
    success_rate = round(1.0 * cnt_success / num_steps, 4)
    #
    if input_tester['file_save'] != None:
        print "saving results ... "
        assert('.pkl' in input_tester['file_save'])
        with open(input_tester['file_save'], 'wb') as f:
            pickle.dump(bs_results, f)
    else:
        print "No need to save results"
    #
    print "the # of paths in this map is : ", (num_steps, name_map)
    print "the success_rate is : ", success_rate
    #
    print "finish testing !!! "


def test_model_ensemble(input_tester):
    '''
    this function is called to test ensemble model
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(12345)
    #

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcess(
        path_rawdata=input_tester['path_rawdata']
    )
    #
    #TODO: build another data process for Greedy search, i.e., gs
    ##
    bs_settings = {
        'size_beam': 1, # greedy search
        'set_path_model': input_tester['set_path_model'],
        #'trained_model': None,
        'dim_lang': data_process.dim_lang,
        'map': data_process.maps[
            data_process.map2idx[input_tester['map_test']]
        ]
    }
    #
    #TODO: build the model
    print "building model ... "
    #
    #
    name_map = input_tester['map_test']
    #
    cnt_success = 0
    num_steps = len(
        data_process.dict_data['dev'][name_map]
    ) + len(
        data_process.dict_data['train'][name_map]
    )
    #
    bs = beam_search.BeamSearchNeuralWalkerEnsemble(
        bs_settings
    )
    #
    bs_results = []
    #
    for idx_data, data in enumerate(data_process.dict_data['dev'][name_map]):
        data_process.process_one_data(
            idx_data, name_map, 'dev'
        )
        bs.set_encoder(
            data_process.seq_lang_numpy,
            data_process.seq_world_numpy
        )
        pos_start, pos_end = data_process.get_pos(
            idx_data, name_map, 'dev'
        )
        bs.init_beam(
            numpy.copy(pos_start), numpy.copy(pos_end)
        )
        bs.search_func()
        #
        if bs.check_pos_end():
            cnt_success += 1
        #
        result = {
            'path_ref': data['cleanpath'],
            'path_gen': bs.get_path(),
            'success': bs.check_pos_end(),
            'pos_current': bs.finish_list[0]['pos_current'],
            'pos_destination': bs.finish_list[0]['pos_destination']
        }
        bs_results.append(result)
        #
        bs.refresh_state()
        #
    #
    #
    for idx_data, data in enumerate(data_process.dict_data['train'][name_map]):
        data_process.process_one_data(
            idx_data, name_map, 'train'
        )
        bs.set_encoder(
            data_process.seq_lang_numpy,
            data_process.seq_world_numpy
        )
        pos_start, pos_end = data_process.get_pos(
            idx_data, name_map, 'train'
        )
        bs.init_beam(
            numpy.copy(pos_start), numpy.copy(pos_end)
        )
        bs.search_func()
        #
        if bs.check_pos_end():
            cnt_success += 1
        #
        result = {
            'path_ref': data['cleanpath'],
            'path_gen': bs.get_path(),
            'success': bs.check_pos_end(),
            'pos_current': bs.finish_list[0]['pos_current'],
            'pos_destination': bs.finish_list[0]['pos_destination']
        }
        bs_results.append(result)
        #
        #
        bs.refresh_state()
        ##
    #
    #
    success_rate = round(1.0 * cnt_success / num_steps, 4)
    #
    if input_tester['file_save'] != None:
        print "saving results ... "
        assert('.pkl' in input_tester['file_save'])
        with open(input_tester['file_save'], 'wb') as f:
            pickle.dump(bs_results, f)
    else:
        print "No need to save results"
    #
    print "the # of paths in this map is : ", (num_steps, name_map)
    print "the success_rate is : ", success_rate
    #
    print "finish testing !!! "
