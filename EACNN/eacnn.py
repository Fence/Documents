#coding:utf-8
import re, os, time, ipdb
import pickle
import argparse
import mysql.connector
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from gensim.models import KeyedVectors
from keras import backend as K
from keras.models import Sequential
from keras.layers import Merge
from keras.layers import Dense, Flatten, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
from utils import ten_fold_split_idx, index2data

import sys
reload(sys)
sys.setdefaultencoding('gb18030')

class ActionClassifier:
    def __init__(self, args, outfile):
        self.outfile = outfile
        self.actionDB = args.actionDB
        self.max_text_num = args.max_text_num
        self.test_text_num = args.test_text_num
        self.vec_size = args.vec_size
        self.words_num = args.words_num
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.train_repeat = args.train_repeat
        self.save_model = args.save_model
        self.verbose = args.verbose
        self.valid_rate = args.valid_rate
        self.opt = args.optimizer
        self.context_num = args.context_num
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=1)
      

    def build_model_cnn(self, jj):
        print '\n-----Building model %d...' % jj
        n_gram_flag = 0
        model_shapes = []
        input_shape = (1, self.words_num, self.vec_size)
        model = Sequential()
        if n_gram_flag:
            model.add(Convolution2D(32, 2, self.vec_size, border_mode='valid', input_shape=input_shape))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(Convolution2D(32, 3, 1, border_mode='valid'))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(Convolution2D(32, 4, 1, border_mode='valid'))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(Convolution2D(32, 5, 1, border_mode='valid'))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(MaxPooling2D(pool_size=(self.words_num - 10, 1)))
            model_shapes.append(model.output_shape)
        else:
            model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(Convolution2D(32, 3, 3, border_mode='valid'))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(Convolution2D(32, 3, 3, border_mode='valid'))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
            model_shapes.append(model.output_shape)

        model.add(Flatten())
        model_shapes.append(model.output_shape)

        model.add(Dense(256, activation='relu'))
        model_shapes.append(model.output_shape)

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(optimizer=self.opt, loss='binary_crossentropy', metrics=['accuracy'])

        if jj == 0:
            for s in model_shapes:
                print s
        self.model =  model


    def build_merged_cnn(self, jj):
        print '\n-----Building model %d...\n' % jj
        self.filter_num = 32
        input_texts = (1, self.words_num, self.vec_size)
        bi_gram = Sequential()
        bi_gram.add(Convolution2D(self.filter_num, 2, self.vec_size - 1, 
                        border_mode='valid', input_shape=input_texts))
        bi_gram.add(Activation('relu'))
        bi_gram.add(MaxPooling2D(pool_size=(self.words_num - 1, 1)))

        tri_gram = Sequential()
        tri_gram.add(Convolution2D(self.filter_num, 3, self.vec_size - 1, 
                        border_mode='valid', input_shape=input_texts))
        tri_gram.add(Activation('relu'))
        tri_gram.add(MaxPooling2D(pool_size=(self.words_num - 2, 1)))

        four_gram = Sequential()
        four_gram.add(Convolution2D(self.filter_num, 4, self.vec_size - 1, 
                        border_mode='valid', input_shape=input_texts))
        four_gram.add(Activation('relu'))
        four_gram.add(MaxPooling2D(pool_size=(self.words_num - 3, 1)))

        five_gram = Sequential()
        five_gram.add(Convolution2D(self.filter_num, 5, self.vec_size - 1, 
                        border_mode='valid', input_shape=input_texts))
        five_gram.add(Activation('relu'))
        five_gram.add(MaxPooling2D(pool_size=(self.words_num - 4, 1)))

        merged = Merge([bi_gram, tri_gram, four_gram, five_gram], mode='concat')

        model = Sequential()
        model.add(merged)
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(optimizer=self.opt, loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model


    def build_model_mlp(self, jj):
        print '\n-----Building model %d...' % jj
        dim = (self.context_num * 2 + 1 ) * self.vec_size
        print 'model input dimension: ', dim
        model_shapes = []
        model = Sequential()
        model.add(Dense(self.vec_size, input_dim=dim, init='uniform', activation='relu'))
        model_shapes.append(model.output_shape)
        model.add(Dense(1))
        model_shapes.append(model.output_shape)
        model.add(Activation('sigmoid'))
        model.compile(optimizer=self.opt, loss='binary_crossentropy', metrics=['accuracy'])
        if jj == 0:
            for s in model_shapes:
                print s
        self.model = model


    def train_one(self, x, label):
        return self.model.fit(x, label, 
            batch_size = self.batch_size, 
            nb_epoch = self.epochs,
            verbose = self.verbose, 
            callbacks = [self.early_stopping], 
            validation_split = self.valid_rate, 
            shuffle = True)



    def test_one(self, t_x):
        return self.model.predict_classes(t_x, 
            batch_size = self.batch_size, 
            verbose = self.verbose)


    def save_model(jj):
        print '\nSaving model: %s_%d.h5 ...' % (self.save_model, jj)
        self.model.save("%s_%d.h5" % (self.save_model, jj))

 
    
def record_history(hist, outfile, pr, t_text_words, t_label):
    for i in xrange(len(hist)):
        outfile.write('\n\nclassifier %d\n' % i)
        for h in hist[i].history:
            outfile.write(h+':\n')
            for e in hist[i].history[h]:
                outfile.write(str(e)+'\n')
            outfile.write('\n\n')

    for i in xrange(len(pr)):
        for j in xrange(len(pr[i])):
            outfile.write('\n%s    %d    %d' % 
                (t_text_words[i][j], t_label[i][j], pr[i][j])) 


def compute_result(outfile, pr, t_text_len, t_label):
    outfile.write('\n\npredicted results:\n')
    total_actions = 0
    total_right_actions = 0
    total_tag_actions = 0
    total_precision = 0
    total_recall = 0
    total_f1_value = 0
    total_tag_right = 0
    total_tag_wrong = 0
    accuracy = 0

    outfile.write('\nt_text_len:'+str(t_text_len))
    #ipdb.set_trace()
    for i in xrange(len(pr)):
        actions = sum(t_label[i][:len(pr[i])])
        right_actions = 0
        tag_actions = 0
        precision = 0
        recall = 0
        f_measure = 0
        right_tag = 0
        wrong_tag = 0
        for j in xrange(len(pr[i])):#xrange(t_text_len[i]):#
            if pr[i][j] == 1:
                tag_actions += 1
                if pr[i][j] == t_label[i][j]:
                    right_actions += 1
            if pr[i][j] == t_label[i][j]:
                right_tag += 1
            else:
                wrong_tag += 1

        tag_right_rate = float(right_tag)/len(pr[i])

        if tag_actions == 0:
            precision = 0
            recall = 0
        else:
            recall = float(right_actions)/actions
            precision = float(right_actions)/tag_actions
        if (recall+precision) == 0:
            f_measure = 0
        else:
            f_measure = 2*recall*precision/(recall+precision)
        '''    
        outfile.write('\n\ntext_num: %d' % i)
        #outfile.write('\npr:' + str(pr[i]))
        #outfile.write('\nt_label:' + str(t_label[i][:len(pr[i])]))
        outfile.write('\nactions: %d' % actions)
        outfile.write('\nright_actions: %d' % right_actions)
        outfile.write('\ntag_actions: %d' % tag_actions)
        outfile.write('\nrecall: %f' % recall)
        outfile.write('\nprecision: %f' % precision)
        outfile.write('\nf_measure: %f' % f_measure )
        print '\n\ntext_num: %d' % i
        print 'actions: %d' % actions
        print 'right_actions: %d' % right_actions
        print 'tag_actions: %d' % tag_actions
        print 'recall: %f' % recall
        print 'precision: %f' % precision
        print 'f_measure: %f' % f_measure 

        outfile.write('\nright_tag: %d\nwrong_tag: %d\ntag_right_rate: %f\n' %
            (right_tag,wrong_tag,tag_right_rate))
        print('\nright_tag: %d\nwrong_tag: %d\ntag_right_rate: %f\n' %
            (right_tag,wrong_tag,tag_right_rate))
        '''

        total_tag_right += right_tag
        total_tag_wrong += wrong_tag
        total_actions += actions
        total_right_actions += right_actions
        total_tag_actions += tag_actions


    accuracy = float(total_tag_right)/(sum([len(pri) for pri in pr]))       

    if total_tag_actions == 0:
        total_precision = 0
        total_recall = 0
    else:
        total_recall = float(total_right_actions)/total_actions
        total_precision = float(total_right_actions)/total_tag_actions
    if (total_recall+total_precision) == 0:
        total_f1_value = 0
    else:
        total_f1_value = 2*total_recall*total_precision/(total_recall+total_precision)
    outfile.write('\n\ntotal_actions: %d' % total_actions)
    outfile.write('\ntotal_right_actions: %d' % total_right_actions)
    outfile.write('\ntotal_tag_actions: %d' % total_tag_actions)
    outfile.write('\ntotal_recall: %f' % total_recall)
    outfile.write('\ntotal_precision: %f' % total_precision)
    outfile.write('\ntotal_f_measure: %f\n' % total_f1_value )
    print '\ntotal_actions: %d' % total_actions
    print 'total_right_actions: %d' % total_right_actions
    print 'total_tag_actions: %d' % total_tag_actions
    print 'total_recall: %f' % total_recall
    print 'total_precision: %f' % total_precision
    print 'total_f1_value: %f' % total_f1_value 

    outfile.write('\ntotal_tag_right: %d\ttotal_tag_wrong: %d\taccuracy: %f\n' %
        (total_tag_right,total_tag_wrong,accuracy))
    print('\ntotal_tag_right: %d\ttotal_tag_wrong: %d\taccuracy: %f\n' %
        (total_tag_right,total_tag_wrong,accuracy))
    return total_precision, total_recall, total_f1_value


def read_data(args, train_flag):
    w_model = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    db = mysql.connector.connect(user='fengwf',password='123',database='test')
    cur = db.cursor()
    result =  []
    for ind,a in enumerate(args.actionDB):
        if train_flag:
            for i in xrange(args.max_text_num[ind] - args.test_text_num):
                get_data = "select * from " + a + ' where text_num = ' + str(i)
                cur.execute(get_data)
                result.append(cur.fetchall())
        else:
            for i in xrange(args.max_text_num[ind] - args.test_text_num, 
                args.max_text_num[ind]):
                get_data = "select * from " + a + ' where text_num = ' + str(i)
                cur.execute(get_data)
                result.append(cur.fetchall())
    
    text_vec = []
    text_label = []
    text_len = []
    text_words = []
    for j in xrange(len(result)):
        words = []
        tags = []
        for k in xrange(len(result[j])):
            sent_lower = result[j][k][2][0].lower() + result[j][k][2][1:]  
            words_of_sent = re.split(r' ',sent_lower)
            temp_tags_of_sent = re.split(r' ',result[j][k][3])
            tags_of_sent = [int(t) for t in temp_tags_of_sent]  
            words.extend(words_of_sent)
            tags.extend(tags_of_sent)
        
        N = len(words) / args.words_num
        M = len(words) % args.words_num
        if M > 0:
            N += 1
        for n in xrange(N):
            start_idx = n * args.words_num
            word_vec = []
            word_label = []
            tmp_words = []
            for l in xrange(args.words_num):
                if start_idx + l < len(words):
                    w = words[start_idx + l]
                    word_label.append(tags[start_idx + l])
                    tmp_words.append(w)
                    if w in w_model.vocab:
                        temp_vec = w_model[w]
                    else:
                        temp_vec = np.zeros(args.vec_size)
                else:
                    tmp_words.append('<NULL>')
                    #words.append('null')
                    word_label.append(0)
                    temp_vec = np.zeros(args.vec_size)
                word_vec.append(temp_vec)
            
            text_vec.append(word_vec)
            text_label.append(word_label)
            text_words.append(tmp_words)
            if n == N - 1:
                text_len.append(len(words) - n*args.words_num)
            else:
                text_len.append(args.words_num)
    print 'texts num: %d\ttext matrix num: %d\n' % (len(result), len(text_vec))
    return np.asarray(text_vec), np.asarray(text_label), text_len, text_words


def data2vec(data, t_data, args, jj):
    if args.context_num == 0:
        in_x = data[:,jj]
        in_t_x = t_data[:,jj]
        return in_x, in_t_x
    cn = args.context_num
    wn = cn * 2 + 1

    padding1 = np.zeros([len(data), cn, args.vec_size])
    padding2 = np.zeros([len(t_data), cn, args.vec_size])
    tmp_data = np.concatenate((padding1, data, padding1), axis=1)
    tmp_t_data = np.concatenate((padding2, t_data, padding2), axis=1)

    in_x = tmp_data[:, jj: jj+wn]
    in_t_x = tmp_t_data[:, jj: jj+wn]
    in_x = in_x.reshape((-1, wn*args.vec_size))
    in_t_x = in_t_x.reshape((-1, wn*args.vec_size))
    
    return in_x, in_t_x


def main(args, outfile):
    #ipdb.set_trace()
    start = time.time()
    raw_data, raw_label, train_text_len, train_text_words = read_data(args, 1)
    print '\nraw_data.shape: {}\traw_label.shape: {}'.format(raw_data.shape, raw_label.shape)
    data = []
    label = []
    for tr in xrange(args.train_repeat):
        data.extend(raw_data)
        label.extend(raw_label)
    data = np.asarray(data)
    label = np.asarray(label)
    print 'data.shape: {}\tlabel.shape: {}\n'.format(data.shape, label.shape)
    x = np.zeros((len(data), 1, args.words_num, args.vec_size))
    x = data.reshape((len(data), 1, args.words_num, args.vec_size))

    t_data, t_label, t_text_len, t_text_words = read_data(args, 0)
    t_x = t_data.reshape((len(t_data), 1, args.words_num, args.vec_size))

    hist = []
    pr = []
    for jj in xrange(args.words_num):
        merged_cnn_flag = False
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_rate
        set_session(tf.Session(config=config))
        AC = ActionClassifier(args, outfile)
        print 'args.text_name: %s' % args.text_name
        if args.model_type == 'cnn':
            AC.build_model_cnn(jj)
            position1 = np.zeros([len(data), 1, args.words_num, args.vec_size])
            position2 = np.zeros([len(t_data), 1, args.words_num, args.vec_size])
            for idx in xrange(args.words_num):
                position1[:,0,idx] = np.abs(idx - jj) / 10.0
                position2[:,0,idx] = np.abs(idx - jj) / 10.0
            in_x = x + position1
            in_t_x = t_x + position2
            #ipdb.set_trace()
        elif args.model_type == 'mcnn':
            AC.build_merged_cnn(jj)
            position1 = np.zeros([len(data), 1, args.words_num, args.vec_size])
            position2 = np.zeros([len(t_data), 1, args.words_num, args.vec_size])
            for idx in xrange(args.words_num):
                position1[:,0,idx] = np.abs(idx - jj) / 10.0
                position2[:,0,idx] = np.abs(idx - jj) / 10.0
            in_x = x + position1
            in_t_x = t_x + position2
            print '-----Training model %d...' % jj
            temp_hist = AC.model.fit([in_x for k in range(4)], label[:,jj], batch_size=args.batch_size, 
                epochs=args.epochs, verbose=args.verbose, callbacks=[AC.early_stopping],
                validation_split=args.valid_rate, shuffle=True)
            print '-----Testing model %d...' % jj
            temp_pr = AC.model.predict_classes([in_t_x for k in range(4)], 
                batch_size=args.batch_size, verbose=args.verbose)
            merged_cnn_flag = True
        else:
            AC.build_model_mlp(jj)
            in_x, in_t_x = data2vec(data, t_data, args, jj)

        if not merged_cnn_flag:
            print '-----Training model %d...' % jj
            temp_hist = AC.train_one(in_x, label[:,jj])
            print '-----Testing model %d...' % jj
            temp_pr = AC.test_one(in_t_x)
        hist.append(temp_hist)
        if len(pr) == 0:
            pr = temp_pr 
        else:
            pr =  np.concatenate([pr, temp_pr],axis=1)
        #AC.save_model(jj)
        tf.reset_default_graph()
    #record_history(hist, outfile, pr, t_text_words, t_label)
    pre, rec, f1 = compute_result(outfile, pr, t_text_len, t_label)

    end = time.time()
    outfile.write("\nTime cost %ds\n" % (end - start))
    print "\n\nTime cost %ds" % (end - start)
    return f1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='/home/fengwf/Documents/mymodel-new-5-50', help='')
    parser.add_argument("--text_name", default='tb_results/tb5/test_mlp_context0.txt', help='')
    parser.add_argument("--save_model", default='models/tb0_model', help='')
    parser.add_argument("--actionDB", default='tag_actions5', help='')
    parser.add_argument("--max_text_num", default='35', help='')
    parser.add_argument("--test_text_num", type=int, default=4, help='')
    parser.add_argument("--words_num", type=int, default=500, help='')
    parser.add_argument("--vec_size", type=int, default=50, help='')
    parser.add_argument("--batch_size", type=int, default=8, help='')
    parser.add_argument("--epochs", type=int, default=100, help='')
    parser.add_argument("--train_repeat", type=int, default=1, help='')
    parser.add_argument("--verbose", type=int, default=0, help='')
    parser.add_argument("--valid_rate", type=float, default=0.2, help='')
    parser.add_argument("--model_type", default='mlp', help='')
    parser.add_argument("--context_num", type=int, default=0, help='')
    parser.add_argument("--optimizer", default='rmsprop', help='')
    parser.add_argument("--gpu_rate", type=float, default=0.4, help='')

    args = parser.parse_args()
    K.set_image_data_format('channels_first')
    #print K.image_data_format()
    actionDBs = ['tag_actions', 'tag_actions1', 'tag_actions2', 'tag_actions3', 'tag_actions4',
                'tag_actions5', 'tag_actions6']
    mtns = [64, 52, 33, 54, 111, 35, 43]
    ttns = [7, 6, 4, 6, 11, 4, 5]
    for tb in xrange(7):
        args.actionDB = actionDBs #[actionDBs[tb]]
        args.max_text_num = mtns #[mtns[tb]]
        args.test_text_num = 7 #ttns[tb]
        f1_values = []
        for ct in xrange(6):
            args.text_name = 'tb_results/tb%d/mlp_context%d.txt' % (tb, ct)
            args.context_num = ct
            #args.actionDB = args.actionDB.split()
            #args.max_text_num = [int(mtn) for mtn in args.max_text_num.split()]
            with open(args.text_name,'w+') as outfile:
                print(str(args)+'\n')
                outfile.write(str(args)+'\n')
                f1 = main(args, outfile)
                f1_values.append(f1)
                print('\nf1_values: {}\navg_f1: {}\n'.format(
                                f1_values, sum(f1_values)/len(f1_values)))
                outfile.write('\nf1_values: {}\navg_f1: {}\n'.format(
                                f1_values, sum(f1_values)/len(f1_values)))
