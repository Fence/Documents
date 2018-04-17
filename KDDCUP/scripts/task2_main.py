import re
import os
import time
import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras import optimizers
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor  
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data_processing import DataProcessing

def build_model(args):
    if args.use_conv:
        model = Sequential()
        for fts in args.layers[0]:
            model.add(Conv2D(fts, (1, 3), strides=args.cnn_str, activation=args.act, 
                input_shape=(1, args.in_dim, 1)))
        #model.add(MaxPooling2D(pool_size=(1,3)))
        #model.add(Conv2D(args.layers[1], (1, 3), strides=(1, 2), activation=args.act))
        model.add(MaxPooling2D(pool_size=(1,3)))
        #model.add(Dropout(0.25))
        model.add(Flatten())
        for dim in args.layers[1]:
            model.add(Dense(dim, activation=args.act))
        model.add(Dense(1))
        print model.summary()
        opt = get_optimizer(args)
        model.compile(optimizer=opt, loss=args.loss, metrics=['acc'])

    else:
        model = Sequential()
        for i in xrange(len(args.layers)):
            if i == 0:
                model.add(Dense(args.layers[i], input_dim=args.in_dim, 
                    kernel_initializer=args.init, activation=args.act))
            elif i == len(args.layers) - 1:
                 model.add(Dense(args.layers[i], kernel_initializer=args.init))
            else:
                model.add(Dense(args.layers[i], kernel_initializer=args.init, activation=args.act))
        print model.summary()
        opt = get_optimizer(args)
        model.compile(optimizer=opt, loss=args.loss, metrics=['acc'])

    return model


def get_optimizer(args):
    if not args.paras:
        return args.opt
    if args.opt == 'sgd':
        opt = optimizers.SGD(lr=args.paras['lr'], decay=args.paras['decay'],
            momentum=args.paras['momentum'], nesterov=args.paras['nesterov'])
    elif args.opt == 'rmsprop':
        opt = optimizers.RMSprop(lr=args.paras['lr'], rho=args.paras['rho'],
            epsilon=args.paras['epsilon'])
    elif args.opt == 'adagrad':
        opt = optimizers.Adagrad(lr=args.paras['lr'], epsilon=args.paras['epsilon'])
    elif args.opt == 'adadelta':
        opt = optimizers.Adadelta(lr=args.paras['lr'], rho=args.paras['rho'],
            epsilon=args.paras['epsilon'])
    elif args.opt == 'adam':
        opt = optimizers.Adam(lr=args.paras['lr'], beta_1=args.paras['beta_1'],
            beta_2=args.paras['beta_2'], epsilon=args.paras['epsilon'])
    elif args.opt == 'adamax':
        opt = optimizers.Adamax(lr=args.paras['lr'], beta_1=args.paras['beta_1'],
            beta_2=args.paras['beta_2'], epsilon=args.paras['epsilon'])
    elif args.opt == 'nadam':
        opt = optimizers.Nadam(lr=args.paras['lr'], beta_1=args.paras['beta_1'],
            beta_2=args.paras['beta_2'], epsilon=args.paras['epsilon'],
            schedule_decay=args.paras['schedule_decay'])
    else:
        opt = 'rmsprop'
    return opt


def combine_parameter(dictionay):
    keys = dictionay.keys()
    keys.sort()
    #lists = [l for k, l in dictionay.items()]
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
    # 4
    opts['rmsprop'] = {'lr': [0.001, 0.0025, 0.005, 0.0005, 0.01], \
                        'rho': [0.9], \
                        'epsilon': [1e-6]}
    # 1
    opts['adagrad'] = {'lr': [0.01], \
                        'epsilon': [1e-6]}
    # 1
    opts['adadelta'] = {'lr':[1.0], \
                        'rho': [0.95], \
                        'epsilon': [1e-6]}
    # 4
    opts['adam'] = {'lr': [0.001, 0.0025, 0.0005], \
                    'beta_1': [0.9], \
                    'beta_2': [0.999], \
                    'epsilon': [1e-8]}
    # 1
    opts['adamax'] = {'lr': [0.002, 0.001, 0.005, 0.01], \
                    'beta_1': [0.9], \
                    'beta_2': [0.999], \
                    'epsilon': [1e-8]}
    # 1
    opts['nadam'] = {'lr': [0.002, 0.001, 0.005, 0.01], \
                    'beta_1': [0.9], \
                    'beta_2': [0.999], \
                    'epsilon': [1e-8], \
                    'schedule_decay': [0.004]}

    loss_func = ['mse', 'mae', 'mape', 'msle', 'squared_hinge', 'hinge']
    #mse: mean_square_error 
    #mae: mean_absolute_error
    #mape: mean_absolute_percentage_error
    #msle: mean_square_logarithmic_error
    return opts, loss_func



def train_model(args, model, x_train, y_train, xtt, ytt):
    if args.use_conv:
        print x_train.shape, y_train.shape, xtt.shape, ytt.shape
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], 1))
        xtt = np.reshape(xtt, (xtt.shape[0], 1, xtt.shape[1], 1))
        print x_train.shape, y_train.shape, xtt.shape, ytt.shape
    #assert 1==0
    if args.callback == 'early_stop':
        callback = EarlyStopping(monitor='val_loss', patience=args.patience)
    elif args.callback == 'checkpoint':
        callback = checkpointer = ModelCheckpoint(
            filepath=args.weight_dir, verbose=1, save_best_only=True)
    if args.valid:
        print '\n----- valid = 1 -----\n'
        record = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.b_size,
            verbose=args.verbo, callbacks=[callback], validation_data=(xtt,ytt))
    else:
        print '\n----- valid = 0 -----\n'
        record = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.b_size,
            verbose=args.verbo, callbacks=[callback], validation_split=args.v_split)

    with open(args.record_dir, 'w+') as f:
        for h in record.history:
            f.write(h+'\n')
            for e in record.history[h]:
                f.write(str(e)+'\n')
            f.write('\n')
    return record


def compute_MAPE(result, y_test):
    #result = np.array(result)
    #result = result.reshape((result.shape[0],result.shape[1]))
    assert len(result) == len(y_test)
    #print 'len(result) == len(y_test) == ',len(y_test)
    #print 'result.shape: ',result.shape
    #print 'y_test.shape: ',y_test.shape
    score = 0
    for tg in xrange(len(result)):
        err = abs(y_test[tg] - result[tg])/y_test[tg]
        #print 'sum(err) = %f'%sum(err)
        #err = [abs((y_test[tg][i]-result[tg][i])/y_test[tg][i]) \
        #for i in xrange(len(result[tg]))]
        #print len(err),err
        #assert 1==0
        temp = sum(err)/float(len(result[tg]))        
        score += temp
        #print 'temp = %f \t score = %f'%(temp,score)
    MAPE = score/float(len(result))
    return MAPE



def main(args, x_train, y_train, x_test, y_test):
    #pdb.set_trace()
    model = build_model(args)
    #assert 1==0

    mms = preprocessing.MinMaxScaler()
    #xtr = mms.fit_transform(x_train)
    #xtr = preprocessing.scale(x_train)
    xtr = x_train
    xtt = x_test.reshape((x_test.shape[0]*x_test.shape[1],x_test.shape[2]))
    ytt = y_test.reshape((y_test.shape[0]*y_test.shape[1],1))
    #print xtt.shape, ytt.shape
    #assert 1==0

    record = train_model(args, model, xtr, y_train, xtt, ytt)
    if os.path.isfile(args.weight_dir):
        model.load_weights(args.weight_dir, by_name=False) # load the best model
    for i in xrange(len(x_test)):
        #xte = mms.fit_transform(x_test[i])
        #xte = preprocessing.scale(x_test[i])
        xte = x_test[i]
        if args.use_conv:
            xte = np.reshape(xte, (xte.shape[0], 1, xte.shape[1], 1))
        tmp = model.predict(xte, batch_size=1)
        tmp = tmp.reshape((tmp.shape[1],tmp.shape[0]))
        if not i:
            result = tmp
        else:
            result = np.concatenate((result, tmp), axis=0)
        #print result.shape
    #assert 1==0
    MAPE = compute_MAPE(result, y_test)
    with open(args.result_dir, 'w+') as f:
        for i in xrange(len(result)):
            f.write('\ntollgate direction pair %d\n'%(i+1))
            f.write('predicted\tfact\n')
            for j in xrange(len(result[i])):
                f.write('%f\t%f\n'%(result[i][j], y_test[i][j]))
        f.write('\nMAPE:%d\n'%MAPE)
    #plot(model, to_file='model.ps')
    #del model
    return MAPE


def change_hyper_parameters(bs, layers, args, opts, debug, wdir):
    best_mape = 100
    best_args = args
    best_count = 0
    count = 0
    MAPE = 100
    inputer = DataProcessing()
    x_train, y_train, x_test, y_test = inputer.combine_data3(args.save_file)
    for b_size in bs:
        for lys in layers:
            args.b_size = b_size
            args.layers = lys

            for k1 in opts.keys():
                op = opts[k1]
                args.opt = k1  # choose an optimizer
                para_list = combine_parameter(op)

                for p in para_list:
                    args.paras = p
                    print 'batch_size = %d\tlayers = %s'%(args.b_size, args.layers)
                    debug.write('\nbatch_size = %d\tepochs = %d\tlayers = %s\n'\
                        %(args.b_size, args.epochs, args.layers))
                    print 'args.opt:\t%s'%args.opt
                    print 'args.paras:\n',args.paras
                    #assert 1==0
                    debug.write('args.opt:\t%s\n'%args.opt)
                    debug.write('args.paras:\n%s\n'%str(args.paras))
                    count += 1
                    args.weight_dir = wdir+'count%d.hdf5'%count
                    MAPE = main(args, x_train, y_train, x_test, y_test)
                    K.clear_session()
                    if best_mape > MAPE:
                        best_mape = MAPE
                        best_args = args
                        best_count = count
                        debug.write('best result:\nbest_mape = %f\
                            best_count = %d\nbest_args = %s\n'%(best_mape, best_count, str(best_args)))
                        print 'best result:\nbest_mape = %f\
                            best_count = %d\nbest_args = %s\n'%(best_mape, best_count, str(best_args))

                    localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                    debug.write('count = %d\n'%count)
                    debug.write('MAPE = %f\n\n'%MAPE)  
                    print 'count = %d'%count
                    print 'MAPE = %f'%MAPE  
                    print 'Current time is:  %s\n\n'%str(localtime)
    return best_mape, best_args, best_count


def change_bs_eps(bs, ep, args, debug):
    best_mape = 100
    best_args = args
    best_count = 0
    count = 0
    inputer = DataProcessing()
    x_train, y_train, x_test, y_test = inputer.combine_data3(args.save_file)
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
            MAPE = main(args, x_train, y_train, x_test, y_test)
            if best_mape > MAPE:
                best_mape = MAPE
                best_args = args
                best_count = count
            localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
            debug.write('count = %d\n'%count)
            debug.write('MAPE = %f\n\n'%MAPE)  
            print 'count = %d'%count  
            print 'MAPE = %f'%MAPE
            print 'Current time is:  %s\n\n'%str(localtime)
    return best_mape, best_args, best_count


def test():
    inputer = DataProcessing()
    x_train, y_train, x_test, y_test = inputer.combine_data3(0)
    #x_train, y_train, x_test, y_test = inputer.combine_data_short(0)
    y_train.shape = (y_train.shape[0],1)
    print x_train.shape, y_train.shape, x_test.shape, y_test.shape
    MAPE = main(args, x_train, y_train, x_test, y_test)
    print 'MAPE = %f'%MAPE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_file', default=0, help='')
    parser.add_argument('--in_dim', type=int, default=50, help='')
    parser.add_argument('--layers', type=list, default=[64,32,16,8,1], help='')
    parser.add_argument('--cnn_str', type=tuple, default=(1,1), help='')
    parser.add_argument('--init', default='normal', help='uniform, normal')
    parser.add_argument('--opt', default='nadam', help='adam, rmsprop, sgd, nadam, adamax')
    parser.add_argument('--paras', type=dict, default={}, help='')
    parser.add_argument('--act', default='relu', help='relu, tanh, sigmoid, softmax')
    parser.add_argument('--loss', default='mape', help='mape, mse, mae, mlse, hinge, squared_hinge')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--b_size', type=int, default=8, help='')
    parser.add_argument('--verbo', type=int, default=2, help='0, 1, 2')
    parser.add_argument('--v_split', type=int, default=0.1, help='')
    parser.add_argument('--valid', default=1, help='offer extra data for validation?')
    parser.add_argument('--callback', default='checkpoint', help='early_stop, checkpoint')
    parser.add_argument('--weight_dir', default='./weights/weights.hdf5', help='')
    parser.add_argument('--patience', type=int, default=3, help='')
    parser.add_argument('--record_dir', default='./results/training_records.txt', help='')
    parser.add_argument('--result_dir', default='./results/result.txt', help='')
    parser.add_argument('--use_conv', type=int, default=1, help='')
    args = parser.parse_args()

    #test()
    #pdb.set_trace()
    with open('./results/debug_dim50_sts11_day111.txt','w+') as debug:
        localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        debug.write('Current time is:  %s\n\n'%str(localtime))
        print 'Current time is:  %s'%str(localtime)
        opts, loss_func = build_opt_loss_dict()
        bs = [4, 8, 10, 16, 32] # batch_size range 8
        ep = [50, 100, 150, 200] # epochs range 4
        #layers = [[args.in_dim, 32, 16, 8, 1], [64, 32, 16, 8, 1], \
        #            [args.in_dim, 32, 16, 1], [args.in_dim, 16, 1], [args.in_dim, 1]] 
        layers = [[[32, 32], [32]], [[16, 16], [32]], [[8, 8], [32]], \
                    [[32, 32], [64, 16]], [[16, 16], [32, 8]], [[8, 8], [16, 4]]]
        # increasing layers
        wdir = './weights1/'
        bm, ba, bc = change_hyper_parameters(bs, layers, args, opts, debug, wdir)
        #bm, ba, bc = change_bs_eps(bs, ep, args, debug)
        debug.write('best result:\nbest_mape = %f\
            best_count = %d\nbest_args = %s\n'%(bm, bc, str(ba)))
        print 'best result:\nbest_mape = %f\
            best_count = %d\nbest_args = %s\n'%(bm, bc, str(ba))

        
        
