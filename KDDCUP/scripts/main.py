import time
import pdb
import argparse
import numpy as np
from keras import backend as K
from data import DataProcessing
from model import KDDCUP


def combine_parameter(dictionay):
    keys = dictionay.keys()
    keys.sort()
    lists = [dictionay[k] for k in keys]
    total = reduce(lambda x, y: x * y, map(len, lists)) if len(lists) > 1 else len(lists[0])
    ret_list = []
    for j in range(total):
        step = total
        temp_item = {}
        for i, key in enumerate(keys):
            l = lists[i]
            step /= len(l)
            temp_item[key] = l[j / step % len(l)]
        ret_list.append(temp_item)

    return ret_list


def build_opt_loss_dict():
    opts = {}
    # 54
    #opts['sgd'] = {'lr': [0.01, 0.025, 0.05], \
    #                'decay': [0.0, 0.5, 0.9], \
    #                'momentum': [0.0, 0.5, 0.9], \
    #                'nesterov': [False, True]}
    # 5
    #opts['rmsprop'] = {'lr': [0.001, 0.0025, 0.005, 0.0005, 0.01], \
    #                    'rho': [0.9], \
    #                    'epsilon': [1e-6]}
    # 3
    opts['adagrad'] = {'lr': [0.01, 0.015, 0.02], \
                        'epsilon': [1e-6]}
    # 4
    opts['adadelta'] = {'lr':[1.0, 0.8, 0.5, 0.2], \
                        'rho': [0.95], \
                        'epsilon': [1e-6]}
    # 4
    opts['adam'] = {'lr': [0.001, 0.0008, 0.0006, 0.0004], \
                    'beta_1': [0.9], \
                    'beta_2': [0.999], \
                    'epsilon': [1e-8]}
    # 5
    opts['adamax'] = {'lr': [0.002, 0.003, 0.004, 0.005, 0.006], \
                    'beta_1': [0.9], \
                    'beta_2': [0.999], \
                    'epsilon': [1e-8]}
    # 6
    opts['nadam'] = {'lr': [0.002, 0.0017, 0.0013, 0.001, 0.0008, 0.0005], \
                    'beta_1': [0.9], \
                    'beta_2': [0.999], \
                    'epsilon': [1e-8], \
                    'schedule_decay': [0.004]}

    loss_func = ['mse', 'mae', 'mape', 'msle', 'squared_hinge', 'hinge']
    #mse: mean_square_error     msle: mean_square_logarithmic_error
    #mae: mean_absolute_error   mape: mean_absolute_percentage_error  
    return opts, loss_func


def change_hyper_parameters(layers, args, opts, debug):
    best_mape = 100;       best_args = args
    best_count = 0;        count = 0;    
    lys_results = [];      opt_results = []
    inputer, data = get_data(args)
    kdd_task2 = KDDCUP(data, inputer)
    save_wdr = args.weight_dir;  save_rdr = args.result_dir
    for lys in layers:
        args.layers = lys
        layer_mape = {'layers':lys, 'layer_mape':[], 'avg_mape':100}

        for k1 in opts.keys():
            op = opts[k1]
            args.opt = k1  # choose an optimizer
            para_list = combine_parameter(op)
            opt_mape = {'opt':k1, 'args':op, 'opt_mape':[], 'avg_mape':100, 'lys':lys}

            for p in para_list:
                args.paras = p
                print '\nlayers = %s\n'%(str(args.layers))
                debug.write('\nlayers = %s\n'%(str(args.layers)))
                print 'args.opt:\t%s'%args.opt
                print 'args.paras:\n',args.paras
                #assert 1==0
                debug.write('args.opt:\t%s\n'%args.opt)
                debug.write('args.paras:\t%s\n'%str(args.paras))
                count += 1
                args.weight_dir = save_wdr + '%0*d'%(4, count)
                args.result_dir = save_rdr + '%0*d'%(4, count)
                kdd_task2.main(args)
                MAPE = kdd_task2.MAPE
                opt_mape['opt_mape'].append(MAPE)
                K.clear_session()
                if best_mape > MAPE:
                    best_mape = MAPE;  best_args = args.__dict__.copy();  best_count = count
                    debug.write('best result:\nbest_mape = %f\tbest_count = %d \
                        \nbest_args = %s\n'%(best_mape, best_count, str(best_args)))
                    print 'best result:\nbest_mape = %f\tbest_count = %d \
                        \nbest_args = %s\n'%(best_mape, best_count, str(best_args))

                localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                debug.write('count = %d\n'%count)
                debug.write('MAPE = %f\n\n'%MAPE)  
                print 'count = %d'%count
                print 'MAPE = %f'%MAPE  
                print 'Current time is:  %s\n\n'%str(localtime)
            avg_opt_mape = sum(opt_mape['opt_mape'])/len(opt_mape['opt_mape'])
            opt_mape['avg_mape'] = avg_opt_mape
            opt_results.append(opt_mape)
            layer_mape['layer_mape'].extend(opt_mape['opt_mape'])

        avg_lys_mape = sum(layer_mape['layer_mape'])/len(layer_mape['layer_mape'])
        layer_mape['avg_mape'] = avg_lys_mape
        lys_results.append(layer_mape)

    for r in lys_results:
        debug.write('layers: %s\t avg_mape: %f\n  layer_mape: %s\n\n'%\
            (str(r['layers']), r['avg_mape'], str(r['layer_mape'])))
        print 'layers: %s\t avg_mape: %f\n  layer_mape: %s\n\n'%\
            (str(r['layers']), r['avg_mape'], str(r['layer_mape']))
    debug.write('\n\n')
    for r in opt_results:
        debug.write('optimizer: %s\t avg_mape: %f\t layers: %s\narguments: %s\nopt_mape: %s\n\n'%\
            (str(r['opt']), r['avg_mape'], str(r['lys']), str(r['args']), str(r['opt_mape'])))
        print 'optimizer: %s\t avg_mape: %f\t layers: %s\narguments: %s\nopt_mape: %s\n\n'%\
            (str(r['opt']), r['avg_mape'], str(r['lys']), str(r['args']), str(r['opt_mape']))

    debug.write('\nbest result:\nbest_mape = %f\tbest_count = %d\
        \nbest_args = %s\n'%(best_mape, best_count, str(best_args)))
    print '\nbest result:\nbest_mape = %f\tbest_count = %d\n \
        \nbest_args = %s\n'%(best_mape, best_count, str(best_args))

    return best_mape, best_args, best_count


def change_hyper_parameters_new(bs, layers, args, opts, debug):
    best_mape = 100;       best_args = args
    best_count = 0;        count = 0;    
    bs_results = [];       lys_results = [];      opt_results = {}
    best_bs_mape = best_lys_mape = best_opt_mape = 100
    inputer, data = get_data(args)
    kdd_task2 = KDDCUP(data, inputer)
    save_wdr = args.weight_dir;  save_rdr = args.result_dir
    for (b_size, t_b_size) in bs:
        bs_mape = {'bs':(b_size, t_b_size), 'bs_mape':[], 'avg_mape':100, 'best_mape':100}
        for lys in layers:
            args.b_size = b_size
            args.t_b_size = t_b_size
            args.layers = lys
            layer_mape = {'lys':lys, 'bs':(b_size, t_b_size), 'layer_mape':[], 'avg_mape':100, 'best_mape':100}

            for k1 in opts.keys():
                op = opts[k1]
                args.opt = k1  # choose an optimizer
                para_list = combine_parameter(op)
                tmp_mape = []
                if k1 not in opt_results.keys():
                    opt_results[k1] = {'args':[op], 'opt_mape':[], 'avg_mape':100, 'best_mape':100,\
                    'lys':[], 'bs':[]}
                
                for p in para_list:
                    args.paras = p
                    print '\nlayers = %s\n'%(str(args.layers))
                    debug.write('\nlayers = %s\n'%(str(args.layers)))
                    print 'args.opt:\t%s'%args.opt
                    print 'args.paras:\n',args.paras
                    debug.write('args.opt:\t%s\n'%args.opt)
                    debug.write('args.paras:\t%s\n'%str(args.paras))
                    count += 1
                    args.weight_dir = save_wdr + '%0*d'%(4, count)
                    args.result_dir = save_rdr + '%0*d'%(4, count)
                    kdd_task2.main(args)
                    MAPE = kdd_task2.MAPE
                    opt_results[k1]['opt_mape'].append(MAPE)
                    opt_results[k1]['bs'].append((b_size, t_b_size))
                    opt_results[k1]['lys'].append(lys)
                    tmp_mape.append(MAPE)
                    K.clear_session()
                    if best_mape > MAPE:
                        best_mape = MAPE;  best_args = args.__dict__.copy();  best_count = count
                        debug.write('best result:\nbest_mape = %f\tbest_count = %d \
                            \nbest_args = %s\n'%(best_mape, best_count, str(best_args)))
                        print 'best result:\nbest_mape = %f\tbest_count = %d \
                            \nbest_args = %s\n'%(best_mape, best_count, str(best_args))
                    if best_bs_mape > MAPE:
                        best_bs_mape = MAPE
                    if best_lys_mape > MAPE:
                        best_lys_mape = MAPE
                    if best_opt_mape > MAPE:
                        best_opt_mape = MAPE

                    localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                    debug.write('count = %d\n'%count)
                    debug.write('MAPE = %f\n\n'%MAPE)  
                    print 'count = %d'%count
                    print 'MAPE = %f'%MAPE  
                    print 'Current time is:  %s\n\n'%str(localtime)
                layer_mape['layer_mape'].extend(tmp_mape)

            avg_lys_mape = sum(layer_mape['layer_mape'])/len(layer_mape['layer_mape'])
            layer_mape['avg_mape'] = avg_lys_mape
            layer_mape['best_mape'] = best_lys_mape
            best_lys_mape = 100
            lys_results.append(layer_mape)
            bs_mape['bs_mape'].extend(layer_mape['layer_mape'])

        avg_bs_mape = sum(bs_mape['bs_mape'])/len(bs_mape['bs_mape'])
        bs_mape['avg_mape'] = avg_bs_mape
        bs_mape['best_mape'] = best_bs_mape
        best_bs_mape = 100
        bs_results.append(bs_mape)

    for r in bs_results:
        debug.write('bs: %s\tbest_mape: %f\tavg_mape: %f\n  bs_mape: %s\n\n'%\
            (str(r['bs']), r['best_mape'], r['avg_mape'], str(r['bs_mape'])))
        print 'bs: %s\tbest_mape: %f\tavg_mape: %f\n  bs_mape: %s\n\n'%\
            (str(r['bs']), r['best_mape'], r['avg_mape'], str(r['bs_mape']))
    debug.write('\n\n')
    for r in lys_results:
        debug.write('layers: %s\tbest_mape: %f\tavg_mape: %f\tbs: %s\n  layer_mape: %s\n\n'%\
            (str(r['lys']), r['best_mape'], r['avg_mape'], str(r['bs']), str(r['layer_mape'])))
        print 'layers: %s\tbest_mape: %f\tavg_mape: %f\tbs: %s\n  layer_mape: %s\n\n'%\
            (str(r['lys']), r['best_mape'], r['avg_mape'], str(r['bs']), str(r['layer_mape']))
    debug.write('\n\n')
    for k,r in opt_results.iteritems():
        r['best_mape'] = min(r['opt_mape'])
        r['avg_mape'] = sum(r['opt_mape'])/len(r['opt_mape'])
        debug.write('optimizer: %s\tbest_mape: %f\tavg_mape: %f\nbs: %s\nlayers: %s\narguments: %s\nopt_mape: %s\n\n'%\
            (str(k), r['best_mape'], r['avg_mape'], str(r['bs']), str(r['lys']), str(r['args']), str(r['opt_mape'])))
        print 'optimizer: %s\tbest_mape: %f\tavg_mape: %f\nbs: %s\nlayers: %s\narguments: %s\nopt_mape: %s\n\n'%\
            (str(k), r['best_mape'], r['avg_mape'], str(r['bs']), str(r['lys']), str(r['args']), str(r['opt_mape']))

    debug.write('\nbest result:\nbest_mape = %f\tbest_count = %d\
        \nbest_args = %s\n'%(best_mape, best_count, str(best_args)))
    print '\nbest result:\nbest_mape = %f\tbest_count = %d\n \
        \nbest_args = %s\n'%(best_mape, best_count, str(best_args))

    return best_mape, best_args, best_count


def change_bs_eps(bs, ep, args, debug):
    best_mape = 100
    best_args = args
    best_count = 0
    count = 0
    inputer, data = get_data(args)
    kdd_task2 = KDDCUP(data, inputer) 
    for b_size in bs:
        for epochs in ep:
            args.b_size = b_size
            args.epochs = epochs
            print 'batch_size = %d\tepochs = %d\tlayers = %s'\
            %(args.b_size, args.epochs, args.layers)
            debug.write('\nbatch_size = %d\tepochs = %d\tlayers = %s\n'\
                %(args.b_size, args.epochs, args.layers))

            count += 1
            args.weight_dir = './weights/count%d.hdf5'%count
            args.result_dir = './results/result%d.txt'%count
            kdd_task2.main(args)
            if best_mape > kdd_task2.MAPE:
                best_mape = kdd_task2.MAPE
                best_args = args
                best_count = count
            localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
            debug.write('count = %d\n'%count)
            debug.write('MAPE = %f\n\n'%kdd_task2.MAPE)  
            print 'count = %d'%count  
            print 'MAPE = %f'%kdd_task2.MAPE
            print 'Current time is:  %s\n\n'%str(localtime)
    return best_mape, best_args, best_count


def get_data(args):
    data = {}
    inputer = DataProcessing(args)
    data['x_train'], data['y_train'], data['x_val'], data['y_val'], \
    data['x_train_dict'], data['x_val_dict'] = inputer.combine_train_data()
    data['x_test'], data['x_test_dict'], data['raw_test_data'] = inputer.init_test_data()

    return inputer, data


def test(args):
    inputer, data = get_data(args)
    kdd_task2 = KDDCUP(data, inputer) 
    kdd_task2.main(args)
    print 'MAPE = %f'%kdd_task2.MAPE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_file', default=0, help='')
    parser.add_argument('--in_dim', type=int, default=54, help='')
    parser.add_argument('--layers', type=list, default=[[16, 16], [32]], help='')
    parser.add_argument('--cnn_str', type=tuple, default=(1,1), help='')
    parser.add_argument('--init', default='normal', help='uniform, normal')
    parser.add_argument('--opt', default='nadam', help='adam, rmsprop, sgd, nadam, adamax')
    parser.add_argument('--paras', type=dict, default={}, help='')
    parser.add_argument('--act', default='relu', help='relu, tanh, sigmoid, softmax')
    parser.add_argument('--loss', default='mape', help='mape, mse, mae, mlse')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--b_size', type=int, default=8, help='')
    parser.add_argument('--t_b_size', type=int, default=8, help='')
    parser.add_argument('--verbo', type=int, default=2, help='0, 1, 2')
    parser.add_argument('--v_split', type=int, default=0.2, help='')
    parser.add_argument('--valid', default=1, help='offer extra data for validation?')
    parser.add_argument('--callback', default='checkpoint', help='early_stop, checkpoint')
    parser.add_argument('--patience', type=int, default=3, help='')
    parser.add_argument('--use_conv', type=int, default=1, help='')
    parser.add_argument('--pool_type', default='max', help='')
    parser.add_argument('--real_test', type=int, default=1, help='')
    parser.add_argument('--refine_val', type=int, default=0, help='')
    parser.add_argument('--mov_time_win', type=int, default=0, help='')
    parser.add_argument('--record_dir', default='', help='')
    # attention
    parser.add_argument('--result_dir', default='./results/test/result', help='')
    parser.add_argument('--weight_dir', default='./weights/test/weight', help='')
    parser.add_argument('--debug_dir', default='./results/records/cnv3_stride1_bs_lr17_dense_lys5_ep100_valid1_maxpool_refine0_bs8_dim54_time_win0.txt', help='')
    args = parser.parse_args()
    
    start = time.time()
    #pdb.set_trace()
    #test(args)
    with open(args.debug_dir, 'w+') as debug:
        localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        debug.write('Current time is:  %s\n\n'%str(localtime))
        print 'Current time is:  %s'%str(localtime)
        opts, loss_func = build_opt_loss_dict()
        bs_list = [4, 8, 10, 16, 32] # batch_size range 8
        bs = []
        for b in bs_list:
            bs.append((b, b))
            bs.append((b, 1))
        print bs
        ep = [50, 100, 150, 200] # epochs range 4
        #layers_dense = [[64, 32, 16, 8, 4, 1], [64, 32, 16, 8, 1], \
        #            [64, 16, 4, 1], [64, 8, 1], [64, 1]] 
        layers_dense = [[64, 32, 16, 8, 1]]
        layers = [[[16, 16], [32]]]#, [[32, 32], [16]], [[32, 32], [32]], [[32, 32], [64]]]
                    #[[32, 32], [64]], [[16, 16], [16, 4]], [[32, 32], [16, 4]]]
        # increasing layers
        #change_hyper_parameters(layers_dense, args, opts, debug)
        change_hyper_parameters_new(bs, layers, args, opts, debug)
        #bm, ba, bc = change_bs_eps(bs_list, ep, args, debug)
        end = time.time()
        print '\ntotal time cost: %ds\n'%(end-start)
        debug.write('\ntotal time cost: %ds\n'%(end-start))
        
        
