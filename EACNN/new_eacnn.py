#coding:utf-8
import re
import os
import time
import ipdb
import pickle
import argparse
import mysql.connector
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from gensim.models import KeyedVectors
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, concatenate, Convolution2D
from keras.layers import Dense, Flatten, Activation, Input, Merge
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session

from utils import ten_fold_split_idx, index2data

import sys
reload(sys)
sys.setdefaultencoding('gb18030')

class ActionClassifier:
    def __init__(self, args):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_rate
        set_session(tf.Session(config=config))
        self.data_dir = args.data_dir
        self.text_name = args.text_name
        self.indices_name = args.indices_name
        self.actionDB = args.actionDB.split()
        self.max_text_num = [int(mtn) for mtn in args.max_text_num.split()]
        self.test_text_num = args.test_text_num
        self.words_num = args.words_num
        self.vec_size = args.vec_size
        self.filter_num = args.filter_num
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.train_repeat = args.train_repeat
        self.optimizer = args.optimizer
        self.verbose = args.verbose

        self.word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=False)
        self.db = mysql.connector.connect(user='fengwf',password='123',database='test')
        self.cur = self.db.cursor()
        
        self.f = args.f
        self.f.write(str(args)+'\n')
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=1)
        self.hist = []
        self.pr = []
      

    def build_merged_CNN(self):
        print '\nBuilding model ...\n'
        input_texts = Input(shape=(self.words_num, self.vec_size, 1))
        bi_gram = Conv2D(self.filter_num, (2, self.vec_size - 1), padding='valid', 
                        activation='relu')(input_texts)
        bi_gram_out = MaxPooling2D((self.words_num - 1, 1), padding='valid')(bi_gram)

        tri_gram = Conv2D(self.filter_num, (3, self.vec_size - 1), padding='valid', 
                        activation='relu')(input_texts)
        tri_gram_out = MaxPooling2D((self.words_num - 2, 1), padding='valid')(tri_gram)

        four_gram = Conv2D(self.filter_num, (4, self.vec_size - 1), padding='valid', 
                        activation='relu')(input_texts)
        four_gram_out = MaxPooling2D((self.words_num - 3, 1), padding='valid')(four_gram)

        five_gram = Conv2D(self.filter_num, (5, self.vec_size - 1), padding='valid', 
                        activation='relu')(input_texts)
        five_gram_out = MaxPooling2D((self.words_num - 4, 1), padding='valid')(five_gram)

        merged = concatenate([bi_gram, tri_gram, four_gram, five_gram], axis=1)
        flatten = Flatten()(merged)
        dense = Dense(1, activation='sigmoid')(flatten)

        model = Model(inputs=input_texts, outputs=dense)
        model.compile(optimizer=self.optimizer, 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

        return model


    def build_merged_CNN2(self):
        print '\nBuilding model ...\n'
        input_texts = (self.words_num, self.vec_size, 1)
        bi_gram = Sequential()
        bi_gram.add(Convolution2D(self.filter_num, (2, self.vec_size - 1), 
                        border_mode='valid', input_shape=input_texts))
        bi_gram.add(Activation('relu'))
        bi_gram.add(MaxPooling2D(pool_size=(self.words_num - 1, 1)))

        tri_gram = Sequential()
        tri_gram.add(Convolution2D(self.filter_num, (3, self.vec_size - 1), 
                        border_mode='valid', input_shape=input_texts))
        tri_gram.add(Activation('relu'))
        tri_gram.add(MaxPooling2D(pool_size=(self.words_num - 2, 1)))

        four_gram = Sequential()
        four_gram.add(Convolution2D(self.filter_num, (4, self.vec_size - 1), 
                        border_mode='valid', input_shape=input_texts))
        four_gram.add(Activation('relu'))
        four_gram.add(MaxPooling2D(pool_size=(self.words_num - 3, 1)))

        five_gram = Sequential()
        five_gram.add(Convolution2D(self.filter_num, (5, self.vec_size - 1), 
                        border_mode='valid', input_shape=input_texts))
        five_gram.add(Activation('relu'))
        five_gram.add(MaxPooling2D(pool_size=(self.words_num - 4, 1)))

        merged = Merge([bi_gram, tri_gram, four_gram, five_gram], mode='concat')

        model = Sequential()
        model.add(merged)
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(optimizer=self.optimizer, 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

        return model


    def build_MLP_model(self, idx):
        #print '\n-----Building model %d...' % idx
        model = Sequential()
        model.add(Dense(self.vec_size, input_dim=self.vec_size, 
                        init='uniform', activation='relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    
    def read_data_for_CNN(self):
        if os.path.exists(self.data_dir):
            data = pickle.load(open(self.data_dir, 'rb'))
        else:
            data = {'texts': [], 'tags': []}
            for idx,a in enumerate(self.actionDB):
                for i in xrange(self.max_text_num[idx]):
                    get_data = 'select * from %s where text_num = %d order by sent_num' % (a, i)
                    print get_data
                    self.cur.execute(get_data)
                    result = self.cur.fetchall()
                    tmp_text = []
                    tmp_tags = []
                    for j in xrange(len(result)):
                        sent = re.split(r' ', result[j][2])
                        tag = [int(t) for t in re.split(r' ', result[j][3])]
                        tmp_text.extend(sent)
                        tmp_tags.extend(tag)

                    data['texts'].append(tmp_text)
                    data['tags'].append(tmp_tags)
            print 'len(data[texts]) = %d' % len(data['texts'])
            with open(self.data_dir, 'wb') as f:
                print 'Try to save file: %s ' % self.data_dir
                pickle.dump(data, f)
                print 'Successfully save file as %s ' % self.data_dir

        data_mat = {'texts': [], 'tags': []}
        for i in tqdm(xrange(len(data['texts']))):
            text_vec = []
            labels = []
            for w in data['texts'][i]:
                if w in self.word2vec.vocab:
                    word_vec = self.word2vec[w]
                else:
                    word_vec = np.zeros(self.vec_size)
                text_vec.append(word_vec)
            text_vec = np.array(text_vec)
            if len(text_vec) < self.words_num:
                text_len = len(text_vec)
                text_vec = np.concatenate((text_vec, 
                    np.zeros([self.words_num - len(text_vec), self.vec_size])))
            else:
                text_len = self.words_num
                text_vec = text_vec[: self.words_num]
            # combine each tag with a text to form a piece of data_mat
            for j in xrange(text_len):
                if data['tags'][i][j] == 1: 
                    position = np.zeros([self.words_num, self.vec_size])
                    for idx in xrange(text_len):
                        position[idx] = (text_len - np.abs(j-idx))/10.0
                    data_mat['texts'].append(text_vec+position)
                    data_mat['tags'].append(data['tags'][i][j])
                else:
                    if np.random.rand() < 0.25:
                        position = np.zeros([self.words_num, self.vec_size])
                        for idx in xrange(text_len):
                            position[idx] = (text_len - np.abs(j-idx))/10.0
                        data_mat['texts'].append(text_vec+position)
                        data_mat['tags'].append(data['tags'][i][j])

        indices = ten_fold_split_idx(len(data_mat['texts']), self.indices_name)
        folds = index2data(indices, data_mat)

        return folds


    def read_data_for_MLP(self, train_flag):
        result =  []
        for ind,a in enumerate(self.actionDB):
            if train_flag:
                for i in range(self.max_text_num[ind] - self.test_text_num):
                    get_data = "select * from " + a + ' where text_num = ' + str(i)
                    print 'get_data',get_data
                    self.cur.execute(get_data)
                    result.append(self.cur.fetchall())
            else:
                for i in range(self.max_text_num[ind] - self.test_text_num, 
                    self.max_text_num[ind]):
                    get_data = "select * from " + a + ' where text_num = ' + str(i)
                    print 'get_data',get_data
                    self.cur.execute(get_data)
                    result.append(self.cur.fetchall())
        
        print 'result.shape',np.asarray(result).shape
        print len(result)
        if train_flag:
            assert len(result) == sum(self.max_text_num) - \
            len(self.actionDB)*self.test_text_num 
        else:
            assert len(result) == len(self.actionDB)*self.test_text_num

        text_vec = []
        text_label = []
        text_len = []
        text_words = []
        for j in range(len(result)):
            words = []
            tags = []
            for k in range(len(result[j])):
                sent_lower = result[j][k][2][0].lower() + result[j][k][2][1:]  
                words_of_sent = re.split(r' ',sent_lower)
                temp_tags_of_sent = re.split(r' ',result[j][k][3])
                tags_of_sent = [int(t) for t in temp_tags_of_sent]
                
                words.extend(words_of_sent)
                tags.extend(tags_of_sent)
            # a long text is splited to several parts
            N = len(result) / self.words_num 
            M = len(result) % self.words_num
            if M > 0: 
                N += 1
            for n in xrange(N):
                word_vec = []
                word_label = []
                start_idx = N * self.words_num
                for l in range(self.words_num):
                    if start_idx + l < len(words):
                        w = words[start_idx + l]
                        word_label.append(tags[start_idx + l])
                        if w in self.word2vec.vocab:
                            temp_vec = self.word2vec[w]
                        else:
                            temp_vec = np.zeros(self.vec_size)
                    else:
                        words.append('NULL')
                        word_label.append(0)
                        temp_vec = np.zeros(self.vec_size)
                    word_vec.append(temp_vec)
                
                text_vec.append(word_vec)
                text_label.append(word_label)
                text_len.append(len(words))
                text_words.append(words)

        print '\nTotal texts: %d\n' % len(text_vec)
        return np.asarray(text_vec), np.asarray(text_label), text_len, text_words


    def train_one(self, x, label):
        temp_hist = self.model.fit(x, label, batch_size=self.batch_size, 
            epochs=self.epochs, verbose=self.verbose, callbacks=[self.early_stopping], 
            validation_split=0.2, shuffle=True)
        self.hist.append(temp_hist)


    def test_one(self, t_x):
        temp_pr = self.model.predict_classes(t_x, batch_size=self.batch_size, verbose=self.verbose)
        if len(self.pr) == 0:
            self.pr = temp_pr 
        else:
            self.pr =  np.concatenate([self.pr, temp_pr],axis=1)

 
    def record_history(self):
        for i in range(len(self.hist)):
            self.f.write('\n\nclassifier %d\n'%i)
            for h in self.hist[i].history:
                self.f.write(h+':\n')
                for e in self.hist[i].history[h]:
                    self.f.write(str(e)+'\n')
                self.f.write('\n\n')

        for i in range(len(self.pr)):
            for j in range(len(self.pr[i])):
                self.f.write('\n%s    %d    %d'%\
                    (self.text_words[i][j],self.t_label[i][j],self.pr[i][j])) 


    def compute_result(self):
        print '\nComputing results ...\n'
        self.f.write('\n\npredicted results:\n')
        total_actions = 0
        total_right_actions = 0
        total_tag_actions = 0
        total_precision = 0
        total_recall = 0
        total_f_measure = 0
        total_tag_right = 0
        total_tag_wrong = 0
        accuracy = 0
        #print '\ntype(pr)',type(self.pr)
        #print 'pr.shape',self.pr.shape
        self.f.write('\nself.text_len:'+str(self.text_len))
        for i in range(len(self.pr)):
            actions = sum(self.t_label[i][:len(self.pr[i])])
            right_actions = 0
            tag_actions = 0
            precision = 0
            recall = 0
            f_measure = 0
            right_tag = 0
            wrong_tag = 0
            for j in range(len(self.pr[i])):#range(self.text_len[i]):#
                if self.pr[i][j] == 1:
                    tag_actions += 1
                    if self.pr[i][j] == self.t_label[i][j]:
                        right_actions += 1
                if self.pr[i][j] == self.t_label[i][j]:
                    right_tag += 1
                else:
                    wrong_tag += 1

            tag_right_rate = float(right_tag)/len(self.pr[i])

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
            self.f.write('\n\ntext_num: %d'%i)
            self.f.write('\npr:'+str(self.pr[i]))
            self.f.write('\nt_label:'+str(self.t_label[i][:len(self.pr[i])]))
            self.f.write('\nactions: %d'%actions)
            self.f.write('\nright_actions: %d'%right_actions)
            self.f.write('\ntag_actions: %d'%tag_actions)
            self.f.write('\nrecall: %f'%recall)
            self.f.write('\nprecision: %f'%precision)
            self.f.write('\nf_measure: %f'%f_measure )
            print '\n\ntext_num: %d'%i
            print 'actions: %d'%actions
            print 'right_actions: %d'%right_actions
            print 'tag_actions: %d'%tag_actions
            print 'recall: %f'%recall
            print 'precision: %f'%precision
            print 'f_measure: %f'%f_measure 

            self.f.write('\nright_tag: %d\n  wrong_tag: %d\n  tag_right_rate: %f\n' % 
                (right_tag,wrong_tag,tag_right_rate))
            print '\nright_tag: %d\n  wrong_tag: %d\n  tag_right_rate: %f\n' % \
                (right_tag,wrong_tag,tag_right_rate)

            total_tag_right += right_tag
            total_tag_wrong += wrong_tag
            total_actions += actions
            total_right_actions += right_actions
            total_tag_actions += tag_actions

        accuracy = float(total_tag_right)/(sum([len(pri) for pri in self.pr]))       

        if total_tag_actions == 0:
            total_precision = 0
            total_recall = 0
        else:
            total_recall = float(total_right_actions)/total_actions
            total_precision = float(total_right_actions)/total_tag_actions
        if (total_recall+total_precision) == 0:
            total_f_measure = 0
        else:
            total_f_measure = 2*total_recall*total_precision/(total_recall+total_precision)
        self.f.write('\n\ntotal_actions: %d'%total_actions)
        self.f.write('\ntotal_right_actions: %d'%total_right_actions)
        self.f.write('\ntotal_tag_actions: %d'%total_tag_actions)
        self.f.write('\ntotal_recall: %f'%total_recall)
        self.f.write('\ntotal_precision: %f'%total_precision)
        self.f.write('\ntotal_f_measure: %f\n'%total_f_measure )
        print '\ntotal_actions: %d'%total_actions
        print 'total_right_actions: %d'%total_right_actions
        print 'total_tag_actions: %d'%total_tag_actions
        print 'total_recall: %f'%total_recall
        print 'total_precision: %f'%total_precision
        print 'total_f_measure: %f'%total_f_measure 

        self.f.write('\ntotal_tag_right: %d total_tag_wrong: %d  accuracy: %f\n' %
            (total_tag_right,total_tag_wrong,accuracy))
        print '\ntotal_tag_right: %d total_tag_wrong: %d  accuracy: %f\n' % \
            (total_tag_right,total_tag_wrong,accuracy)


    def save_model(idx):
        print '\nSaving model: %s ...'%("%s_%d.h5"%(self.save_model, idx))
        self.model.save("%s_%d.h5"%(self.save_model, idx))

        
    def CNN_main(self):
        start = time.time()
        folds = self.read_data_for_CNN()

        for i in xrange(1):#len(folds['train'])):
            print 'Training fold %d' % i
            result_dir = '%s_fold%d.txt' % (self.text_name, i)
            train_data = folds['train'][i]
            valid_data = folds['valid'][i] 
            self.model = self.build_merged_CNN2()

            texts = np.array(train_data['texts'])
            tags = np.array(train_data['tags'])
            texts = texts.reshape((len(texts), self.words_num, self.vec_size, 1))
            #texts = texts.reshape((len(texts), 1, self.words_num, self.vec_size))
            #tags = tags.reshape((len(tags), 1, 1))

            #self.model = self.build_merged_CNN2()
            hist = self.model.fit([texts for k in range(4)], tags, batch_size=self.batch_size, 
                epochs=self.epochs, verbose=self.verbose, callbacks=[self.early_stopping],
                validation_split=0.2, shuffle=True)

            valid_texts = np.array(valid_data['texts'])
            valid_tags = np.array(valid_data['tags'])
            valid_texts = valid_texts.reshape(
                (len(valid_texts), self.words_num, self.vec_size, 1))
            #valid_texts = valid_texts.reshape(
            #    (len(valid_texts), 1, self.words_num, self.vec_size))
            #valid_tags = valid_tags.reshape((len(valid_tags), 1, 1))
            #pred = self.model.predict(valid_texts, batch_size=self.batch_size, verbose=self.verbose)
            pred = self.model.predict_classes([valid_texts for k in range(4)], 
                batch_size=self.batch_size, verbose=self.verbose)
            #ipdb.set_trace()
            total_act = sum(valid_tags)
            right_act = 0
            tagged_act = 0
            precision = 0
            f_measure = 0
            recall = 0

            for j,t in enumerate(pred):
                if t == 1:
                    tagged_act += 1
                    if t == valid_tags[j]:
                        right_act += 1
            if tagged_act > 0:
                precision = float(right_act) / tagged_act
            recall = float(right_act) / total_act
            if precision + recall > 0:
                f_measure = 2 * precision * recall / (precision + recall)

            with open(result_dir, 'w') as f:
                for item in hist.history:
                    print '\n\n',item
                    f.write('\n{}:\n'.format(item))
                    for value in hist.history[item]:
                        print value
                        f.write('{}\n'.format(value))
                print('total_act:%d\ntagged_act:%d\nright_act:%d' % 
                    (total_act, tagged_act, right_act))
                print('recall:%f\nprecision:%f\nf_measure:%f' % 
                    (recall, precision, f_measure))
                f.write('total_act:%d\ntagged_act:%d\nright_act:%d\n' % 
                    (total_act, tagged_act, right_act))
                f.write('recall:%f\nprecision:%f\nf_measure:%f\n' % 
                    (recall, precision, f_measure))
            end = time.time()
            print 'Time cost: %ds' % (end - start)
                

    def MLP_main(self):
        start = time.time()
        train_time = 0
        test_time = 0
        train_start = time.time()
        self.raw_data, self.raw_label, _, __ = self.read_data_for_MLP(1)

        data = []
        label = []
        for tr in range(self.train_repeat):
            data.extend(self.raw_data)
            label.extend(self.raw_label)
        data = np.asarray(data)
        label = np.asarray(label)
        x = data.reshape((len(data), 1, self.words_num, self.vec_size))

        self.t_data, self.t_label, self.text_len, self.text_words = self.read_data_for_MLP(0)
        t_x = self.t_data.reshape((len(self.t_data), 1, self.words_num, self.vec_size))

        for idx in tqdm(xrange(self.words_num)):
            if idx > 0:
                train_start = time.time()
            self.model = self.build_MLP_model(idx)
            #print '-----Training model %d...'%idx
            in_x = data[:,idx]
            self.train_one(in_x, label[:,idx])
            train_end = time.time()
            train_time += (train_end - train_start)

            test_start = time.time()
            #print '-----Testing model %d...'%idx
            in_t_x = self.t_data[:,idx]
            self.test_one(in_t_x)
            test_end = time.time()
            test_time += (test_end - test_start)
        self.record_history()
        self.compute_result()
        tf.reset_default_graph()

        end = time.time()
        self.f.write('\nTrain time: %d,    Test time: %d\n' % (train_time, test_time))
        self.f.write("\nTime cost %ds\n" % (end-start))
        self.f.close()
        print '\nTrain time: %d,    Test time: %d\n' % (train_time, test_time)
        print "\n\nTime cost %ds" % (end-start)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='/home/fengwf/Documents/mymodel5-5-50', help='')
    parser.add_argument("--text_name", default='results/cnn_test', help='')
    parser.add_argument("--data_dir", default='data/databag.pkl', help='')
    parser.add_argument("--indices_name", default='data/10-fold-indices.pkl', help='')
    parser.add_argument("--save_model", default='models/test_model', help='')
    parser.add_argument("--actionDB", default='tag_actions3', help='')
    parser.add_argument("--max_text_num", default='54', help='')
    parser.add_argument("--test_text_num", type=int, default=6, help='')
    parser.add_argument("--filter_num", type=int, default=32, help='')
    parser.add_argument("--words_num", type=int, default=500, help='')
    parser.add_argument("--vec_size", type=int, default=50, help='')
    parser.add_argument("--batch_size", type=int, default=8, help='')
    parser.add_argument("--epochs", type=int, default=200, help='')
    parser.add_argument("--train_repeat", type=int, default=1, help='')
    parser.add_argument("--optimizer", default='adam', help='')
    parser.add_argument("--verbose", type=int, default=0, help='')
    parser.add_argument("--gpu_rate", type=float, default=0.4, help='')

    args = parser.parse_args()
    tables = ['tag_actions5', 'tag_actions6', 'tag_actions', 'tag_actions1', 
                'tag_actions2', 'tag_actions3', 'tag_actions4']
    mtn = [43, 35, 64, 52, 33, 54, 111]
    ttn = [5, 4, 7, 6, 4, 6, 12]
    st = time.time()
    for i in xrange(1):#len(tables)):
        #args.actionDB = tables[i]
        #args.max_text_num = str(mtn[i])
        #args.text_name = 'tb_results/tb%d/test_result' % i
        with open(args.text_name+'_main.txt','w+') as args.f:
            AC = ActionClassifier(args)
            #AC.MLP_main()
            AC.CNN_main()
    et = time.time()
    print '\nTotal time cost: %ds\n' % (et - st)
