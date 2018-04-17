import os
import time
import numpy as np
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor  
#from keras.utils.visualize_util import plot
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint


class KDDCUP():
    def __init__(self, data, inputer):
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_val = data['x_val']
        self.y_val = data['y_val']
        self.x_test = data['x_test']
        self.x_train_dict = data['x_train_dict']
        self.x_val_dict = data['x_val_dict']
        self.x_test_dict = data['x_test_dict']
        self.raw_test_data = data['raw_test_data']
        self.inputer = inputer
        self.MAPE = 100


    def build_model(self, args):
        if args.use_conv:
            self.model = Sequential()
            self.model.add(Conv2D(args.layers[0][0], (1, 3), strides=args.cnn_str, 
                activation=args.act, input_shape=(1, args.in_dim, 1)))
            for fts in args.layers[0][1:]:
                self.model.add(Conv2D(fts, (1, 3), strides=args.cnn_str, activation=args.act))
            if args.pool_type == 'max':
                print '\nmax-pooling\n'
                self.model.add(MaxPooling2D(pool_size=(1,3)))
            else:
                self.model.add(AveragePooling2D(pool_size=(1,3)))
            #self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            for dim in args.layers[1]:
                self.model.add(Dense(dim, activation=args.act))
            self.model.add(Dense(1))

        else:
            self.model = Sequential()
            for i in xrange(len(args.layers)):
                if i == 0:
                    self.model.add(Dense(args.layers[i], input_dim=args.in_dim, 
                        kernel_initializer=args.init, activation=args.act))
                elif i == len(args.layers) - 1:
                     self.model.add(Dense(args.layers[i], kernel_initializer=args.init))
                else:
                    self.model.add(Dense(args.layers[i], kernel_initializer=args.init, 
                        activation=args.act))
        print self.model.summary()
        opt = self.get_optimizer(args)
        self.model.compile(optimizer=opt, loss=args.loss, metrics=['acc'])


    def get_optimizer(self, args):
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


    def train_model(self, args, x_train, y_train, xtt, ytt):
        if args.use_conv:
            x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], 1))
            xtt = np.reshape(xtt, (xtt.shape[0], 1, xtt.shape[1], 1))
            print x_train.shape, y_train.shape, xtt.shape, ytt.shape

        if args.callback == 'early_stop':
            callback = EarlyStopping(monitor='val_loss', patience=args.patience)
        elif args.callback == 'checkpoint':
            callback = checkpointer = ModelCheckpoint(
                filepath=args.weight_dir+'.hdf5', verbose=1, save_best_only=True)
        if args.valid:
            print '\n----- valid = 1 -----\n'
            record = self.model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.b_size,
                verbose=args.verbo, callbacks=[callback], validation_data=(xtt,ytt))
        else:
            print '\n----- valid = 0 -----\n'
            record = self.model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.b_size,
                verbose=args.verbo, callbacks=[callback], validation_split=args.v_split)

        if args.record_dir:
            with open(args.record_dir + '.txt', 'w+') as f:
                f.write('args:\n')
                for k in args.__dict__.keys():
                    f.write(str(k)+':\t'+str(args.__dict__[k])+'\n')
                for h in record.history:
                    f.write('\n'+h+'\n')
                    for e in record.history[h]:
                        f.write(str(e)+'\n')
                    f.write('\n')
        return record


    def test_model(self, args):
        if args.real_test:
            for i in xrange(6): # test 6 itvs
                if i == 0:
                    xte = self.x_test
                    xte_dict = self.x_test_dict
                    #print "xte_dict[0]['hist_v']",xte_dict[0]['hist_v']
                    if args.use_conv:
                        xte = np.reshape(xte, (xte.shape[0], 1, xte.shape[1], 1))
                    pred_y = self.model.predict(xte, batch_size=args.t_b_size)
                    test_result = pred_y
                else:
                    xte, xte_dict = self.inputer.update_test_data(xte_dict, pred_y)
                    #print "xte_dict[0]['hist_v']",xte_dict[0]['hist_v']
                    if args.use_conv:
                        xte = np.reshape(xte, (xte.shape[0], 1, xte.shape[1], 1))
                    pred_y = self.model.predict(xte, batch_size=args.t_b_size)
                    test_result = np.concatenate((test_result, pred_y), axis=1)


        if args.refine_val:
            tmp_x_val = []
            tmp_y_val = [[], [], [], [], [], []]
            tmp_x_dict = [] 
            for k in xrange(len(self.x_val[0])):
                for j in xrange(len(self.x_val)):
                    tmp_y_val[k%6].append(self.y_val[j][k])
                    if k%6 == 0:
                        tmp_x_val.append(self.x_val[j][k])
                        tmp_x_dict.append(self.x_val_dict[j][k])
            for i in xrange(6):
                if i == 0:
                    xte = np.array(tmp_x_val)
                    xte_dict = tmp_x_dict
                    if args.use_conv:
                        xte = np.reshape(xte, (xte.shape[0], 1, xte.shape[1], 1))
                    pred_y = self.model.predict(xte, batch_size=args.t_b_size)
                    val_result = pred_y
                else:
                    xte, xte_dict = self.inputer.update_test_data(xte_dict, pred_y, 1)
                    #print "xte_dict[0]['hist_v']",xte_dict[0]['hist_v']
                    if args.use_conv:
                        xte = np.reshape(xte, (xte.shape[0], 1, xte.shape[1], 1))
                    pred_y = self.model.predict(xte, batch_size=args.t_b_size)
                    val_result = np.concatenate((val_result, pred_y), axis=1)
        else:
            tmp_y_val = []
            for i in xrange(len(self.x_val)):
                xte = self.x_val[i]
                if args.use_conv:
                    xte = np.reshape(xte, (xte.shape[0], 1, xte.shape[1], 1))
                tmp = self.model.predict(xte, batch_size=args.t_b_size)
                tmp = tmp.reshape((tmp.shape[1],tmp.shape[0]))
                if not i:
                    val_result = tmp
                else:
                    val_result = np.concatenate((val_result, tmp), axis=0)
        if tmp_y_val:
            tmp_y_val = np.array(tmp_y_val)
            tmp_y_val = np.reshape(tmp_y_val, (tmp_y_val.shape[1], -1))

        return tmp_y_val, val_result, test_result


    def compute_MAPE(self, result, y_test):
        assert len(result) == len(y_test)
        score = 0
        for tg in xrange(len(result)):
            err = abs(y_test[tg] - result[tg])/y_test[tg]
            temp = sum(err)/float(len(result[tg]))        
            score += temp
        MAPE = score/float(len(result))
        return MAPE


    def main(self, args):
        self.build_model(args)

        xtr = self.x_train
        xtt = self.x_val.reshape((self.x_val.shape[0]*self.x_val.shape[1],self.x_val.shape[2]))
        ytt = self.y_val.reshape((self.y_val.shape[0]*self.y_val.shape[1],1))

        record = self.train_model(args, xtr, self.y_train, xtt, ytt)
        if os.path.isfile(args.weight_dir):
            self.model.load_weights(args.weight_dir, by_name=False) # load the best model
        
        y_val, result, test_result = self.test_model(args)
        if args.refine_val:
            self.MAPE = self.compute_MAPE(result, y_val)
        else:
            self.MAPE = self.compute_MAPE(result, self.y_val)

        #format output for tollgate and direction per time window
        with open(args.result_dir + '.csv', 'w') as fw:
            fw.writelines(','.join(
                ['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
            assert len(self.raw_test_data) == len(test_result)
            for i in xrange(len(test_result[0])):
                for j in xrange(len(test_result)):
                    xd = self.raw_test_data[j]
                    itv = xd['itv'] + i
                    start_time = '%0*d'%(2, itv/3) + ':' + '%0*d'%(2, itv%3*20) + ':00'
                    end_time = '%0*d'%(2, (itv+1)/3) + ':' + '%0*d'%(2, (itv+1)%3*20) + ':00'
                    out_line = ','.join(['"' + xd['gate'] + '"', 
                                 '"['+xd['date']+' '+start_time+',' + xd['date']+' '+end_time+')"',
                                 '"' + xd['dirt'] + '"',
                                 '"' + str(test_result[j][i]) + '"',
                               ]) + '\n'
                    fw.writelines(out_line)

        with open(args.result_dir + '.txt', 'w+') as f:
            f.write('validation MAPE: %f\nargs:\n'%self.MAPE)
            for k in args.__dict__.keys():
                f.write(str(k)+':\t'+str(args.__dict__[k])+'\n')
            f.write('\npredicted results:\n')
            for i in xrange(len(test_result)):
                f.write(str(test_result[i])+'\n')

            f.write('\n\nvalidation results:\n')
            for i in xrange(len(result)):
                f.write('\ntollgate direction pair %d\n'%(i+1))
                f.write('predicted\tfact\n')
                for j in xrange(len(result[i])):
                    if args.refine_val:
                        f.write('%f\t%f\n'%(result[i][j], y_val[i][j]))
                    else:
                        f.write('%f\t%f\n'%(result[i][j], self.y_val[i][j]))
        #plot(self.model, to_file='self.model.ps')
