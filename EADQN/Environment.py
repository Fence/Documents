#coding:utf-8
import re
import ipdb
import time
import pickle
import mysql.connector
import numpy as np
from utils import ten_fold_split_idx, index2data

#create table tag_actions5(text_num int, sent_num int, sent varchar(400), tag_sent varchar(400));
class Environment:
    def __init__(self, args):
        print('Initializing the Environment...')  
        self.batch_act_num = args.batch_act_num
        self.words_num = args.words_num
        self.emb_dim = args.emb_dim
        self.word_dim = args.word_dim
        self.tag_dim = args.tag_dim
        self.action_rate = args.action_rate
        self.action_label = args.action_label
        self.non_action_label = args.non_action_label
        self.reward_base = args.reward_assign
        self.reward_assign = self.reward_base * np.array([2, 1, -1, -2])

        self.ten_fold_valid = args.ten_fold_valid
        self.char_emb_flag = args.char_emb_flag
        self.word2vec = args.word2vec
        self.max_char_len = args.max_char_len
        self.fold_id = args.fold_id
        self.ten_fold_indices = args.ten_fold_indices
        self.actionDB = args.actionDB.split()
        self.max_text_num = [int(t) for t in args.max_text_num.split()]
        self.test_text_num = args.test_text_num
        self.max_data_char_len = 0
        self.max_data_sent_len = 0
        self.char_info()
        self.read_pkl_texts()
        #self.read_database_texts()
        self.context_len = args.context_len
        self.distance = np.abs(np.arange(-self.context_len, self.context_len + 1))
        self.terminal_flag = False
        self.epoch_end_flag = False
        self.obj_dim = args.obj_dim
        self.add_obj_flag = args.add_obj_flag
        

    def char_info(self):
        #chars = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,;!?:._'\"+-*/@#$%"
        chars = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|"
        self.char2idx = {}
        for c in chars:
            self.char2idx[c] = len(self.char2idx)
        return self.char2idx


    def read_pkl_texts(self):
        with open('/home/fengwf/Documents/DRL_data/cooking/labeled_text_data2.pkl', 'r') as f:
            indata = pickle.load(f)
        eas_texts = []
        #ipdb.set_trace()
        for i in xrange(len(indata)):
            eas_text = {}
            eas_text['tokens'] = indata[i]['words']
            eas_text['acts'] = indata[i]['acts']
            eas_text['tags'] = np.ones(len(indata[i]['words']), dtype=np.int32)
            for act_idx in indata[i]['acts']:
                eas_text['tags'][act_idx] = 2

            self.create_matrix(eas_text)
            eas_texts.append(eas_text)
        
        if self.ten_fold_valid:
            eas_indices = ten_fold_split_idx(len(eas_texts), self.ten_fold_indices)
            eas_folds = index2data(eas_indices, eas_texts)
            self.train_data = eas_folds['train'][self.fold_id]
            self.valid_data = eas_folds['valid'][self.fold_id]
            self.train_steps = len(self.train_data) * self.words_num
            self.valid_steps = len(self.valid_data) * self.words_num
        else:
            self.train_data = eas_texts[self.test_text_num: self.max_text_num[0]]
            self.valid_data = eas_texts[: self.test_text_num]
            self.train_steps = self.words_num * (self.max_text_num[0] - self.test_text_num)
            self.valid_steps = self.words_num * self.test_text_num
        self.num_train = len(self.train_data)
        self.num_valid = len(self.valid_data)
        print('\n\ntraining texts: %d\tvalidation texts: %d' % (len(self.train_data), len(self.valid_data)))
        print('max_data_sent_len: %d\tmax_data_char_len: %d' % (self.max_data_sent_len, self.max_data_char_len))
        print('self.train_steps: %d\tself.valid_steps: %d\n\n' % (self.train_steps, self.valid_steps))


    def read_database_texts(self):
        self.db = mysql.connector.connect(user='fengwf',password='123',database='test')
        self.cur = self.db.cursor()
        result = []
        for i in xrange(len(self.max_text_num)):
            for j in xrange(self.max_text_num[i]):
                get_data = "select * from %s where text_num = %d order by sent_num" % (self.actionDB[i], j)
                self.cur.execute(get_data)
                result.append(self.cur.fetchall())

        print('Total texts: %d' % len(result))
        eas_texts = []
        for i in xrange(len(result)):
            eas_text = {}
            eas_text['tokens'] = []
            eas_text['tags'] = []
            for j in xrange(len(result[i])):
                try:
                    tmp_tokens = str(result[i][j][2]).split()
                except Exception as e:
                    continue
                eas_text['tokens'].extend(tmp_tokens)
                eas_text['tags'].extend([int(t)+1 for t in result[i][j][3].split()])
            self.create_matrix(eas_text)
            eas_texts.append(eas_text)
        
        if self.ten_fold_valid:
            eas_indices = ten_fold_split_idx(len(eas_texts), self.ten_fold_indices)
            eas_folds = index2data(eas_indices, eas_texts)
            self.train_data = eas_folds['train'][self.fold_id]
            self.valid_data = eas_folds['valid'][self.fold_id]
            self.train_steps = len(self.train_data) * self.words_num
            self.valid_steps = len(self.valid_data) * self.words_num
        else:
            self.train_data = eas_texts[self.test_text_num: self.max_text_num[0]]
            self.valid_data = eas_texts[: self.test_text_num]
            self.train_steps = self.words_num * (self.max_text_num[0] - self.test_text_num)
            self.valid_steps = self.words_num * self.test_text_num
        self.num_train = len(self.train_data)
        self.num_valid = len(self.valid_data)
        print('\n\ntraining texts: %d\tvalidation texts: %d' % (len(self.train_data), len(self.valid_data)))
        print('max_data_sent_len: %d\tmax_data_char_len: %d' % (self.max_data_sent_len, self.max_data_char_len))
        print('self.train_steps: %d\tself.valid_steps: %d\n\n' % (self.train_steps, self.valid_steps))


    def create_matrix(self, sentence):
        sent_vec = []
        char_vec = []
        for w in sentence['tokens']:
            if len(w) > self.max_data_char_len:
                self.max_data_char_len = len(w)
            if w in self.word2vec.vocab:
                sent_vec.append(self.word2vec[w])
            else:
                sent_vec.append(np.zeros(self.word_dim))
            if len(w) < self.max_char_len:
                w = w + ' '*(self.max_char_len - len(w))
            else:
                w = w[: self.max_char_len]
            char_vec.append([self.char2idx[c] for c in w])
        char_vec = np.array(char_vec, dtype=np.int32)
        sent_vec = np.array(sent_vec)
        pad_len = self.words_num - len(sent_vec)
        if len(sent_vec) > self.max_data_sent_len:
            #ipdb.set_trace()
            self.max_data_sent_len = len(sent_vec)
        if pad_len > 0:
            sent_vec = np.concatenate((sent_vec, np.zeros([pad_len, self.word_dim])))
            char_vec = np.concatenate((char_vec, np.zeros([pad_len, self.max_char_len])))
            sentence['tags'] = np.concatenate((np.array(sentence['tags']), np.ones(pad_len, dtype=np.int32)))
        else:
            sent_vec = sent_vec[: self.words_num]
            char_vec = char_vec[: self.words_num]
            sentence['tags'] = np.array(sentence['tags'])[: self.words_num]
        sentence['sent_vec'] = sent_vec
        sentence['char_vec'] = char_vec
        #ipdb.set_trace()
        tmp_tags = np.zeros([self.words_num, self.tag_dim], dtype=np.int32)
        for i in xrange(self.words_num):
            tmp_tags[i] = sentence['tags'][i]
        sentence['tags'] = tmp_tags


    def restart(self, train_flag, init=False):
        if train_flag:
            if init:
                self.train_text_idx = -1
                self.epoch_end_flag = False
            self.train_text_idx += 1
            if self.train_text_idx >= len(self.train_data):
                self.epoch_end_flag = True
                print('\n\n-----epoch_end_falg = True-----\n\n')
                return
            self.current_text = self.train_data[self.train_text_idx%self.num_train]
            print('restart in env, train_text_idx: %d' % self.train_text_idx)
        else:
            if init:
                self.valid_text_idx = -1
                self.epoch_end_flag = False
            self.valid_text_idx += 1
            if self.valid_text_idx >= len(self.valid_data):
                self.epoch_end_flag = True
                print('\n\n-----epoch_end_falg = True-----\n\n')
                return
            self.current_text = self.valid_data[self.valid_text_idx]
            print('restart in env, valid_text_idx: %d' % self.valid_text_idx)
        #self.current_text.keys() = ['tokens', 'acts', 'tags', 'sent_vec', 'char_vec']
        if not self.add_obj_flag:
            self.text_vec = np.concatenate(
                (self.current_text['sent_vec'], self.current_text['tags']), axis=1)
        else:
            obj_state = np.zeros([self.words_num, self.obj_dim])
            self.text_vec = np.concatenate(
                (obj_state, self.current_text['sent_vec'], self.current_text['tags']), axis=1)
        assert self.text_vec.shape == (self.words_num, self.emb_dim)
        self.state = self.text_vec.copy() # NB!
        self.state[:,self.word_dim:] = 0
        self.terminal_flag = False


    def add_obj_state(self, state2, word_idx):
        obj_state = np.zeros(self.obj_dim)
        for i in xrange(len(state2)):
            if state2[i, -1] == 2:
                obj_state += state2[i, :self.obj_dim]
                #print('\nAdd obj_state\n')
        self.state[word_idx, :self.obj_dim] = obj_state

        
    def act(self, action, steps):
        '''
        Performs action and returns reward
        even num refers to tagging action, odd num refer to non-action
        '''
        #ipdb.set_trace()
        act_str = bin(action)[2:]
        if len(act_str) < self.batch_act_num:
            act_str = '0'*(self.batch_act_num-len(act_str)) + act_str
            act_str = act_str[::-1]
        assert len(act_str) == self.batch_act_num
        #print(act_str)
        bacth_act_reward = 0.0
        for i in range(self.batch_act_num):
            word_idx = steps * self.batch_act_num + i
            # all words of current text are tagged, break
            if word_idx >= len(self.current_text['tokens']):
                self.terminal_flag = True
                #ipdb.set_trace()
                break
            if act_str[i] == '1':  
                self.state[word_idx,self.word_dim:] = self.action_label  
            else: # act_str[i] == '1'
                self.state[word_idx,self.word_dim:] = self.non_action_label    
            t_a_count = 0  #amount of tagged actions 
            for t in self.state[:,-1]:
                if t == self.action_label:
                    t_a_count += 1
            t_a_rate = float(t_a_count)/self.words_num

            label = self.text_vec[word_idx,-1]
            # text_vec is labelled data
            if label == self.state[word_idx,-1]:
                if label == self.action_label:
                    reward = self.reward_assign[0]
                else:
                    reward = self.reward_assign[1] 
            else:
                if self.text_vec[word_idx,-1] == self.non_action_label:
                    reward = self.reward_assign[2]
                else:
                    reward = self.reward_assign[3]
            if t_a_rate <= self.action_rate:
                reward += 5.0 * np.square(t_a_rate) * self.reward_base
            else:
                reward -= 5.0 * np.square(t_a_rate) * self.reward_base
            bacth_act_reward += reward
        return bacth_act_reward


    def getState(self):
        '''
        Gets current text state
        '''
        return self.state


    def isTerminal(self):
        '''
        Returns if tag_actions is done
        if all the words of a text have been tagged, then terminate
        '''
        return self.terminal_flag



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='/home/fengwf/Documents/mymodel-new-5-50', help="")
    parser.add_argument("--max_char_len", type=int, default=20, help="")
    parser.add_argument("--words_num", type=int, default=500, help="")
    parser.add_argument("--word_dim", type=int, default=50, help="")
    parser.add_argument("--emb_dim", type=int, default=100, help="")
    parser.add_argument("--tag_dim", type=int, default=50, help="")
    parser.add_argument("--batch_size", type=int, default=8, help="")
    parser.add_argument("--fold_id", type=int, default=0, help="")
    parser.add_argument("--actionDB", default='tag_actions6', help="")
    parser.add_argument("--max_text_num", default='96', help="")
    parser.add_argument("--reward_assign", type=float, default=1.0, help="")
    parser.add_argument("--action_rate", type=float, default=0.15, help="")
    parser.add_argument("--action_label", type=int, default=2, help="")
    parser.add_argument("--non_action_label", type=int, default=1, help="")
    parser.add_argument("--test", type=int, default=1, help="")
    parser.add_argument("--test_text_num", type=int, default=10, help="")
    parser.add_argument("--char_emb_flag", type=int, default=0, help="")
    parser.add_argument("--ten_fold_valid", type=int, default=1, help="")
    parser.add_argument("--ten_fold_indices", type=str, default='data/cooking-10-fold-indices.pkl', help="")


    args = parser.parse_args()
    from gensim.models import KeyedVectors
    args.word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    env = Environment(args)
    env.read_pkl_texts()
    '''
    env.train_init()
    a = raw_input('Continue?(y/n)').lower()
    while a != 'n':
        env.restart()
        a = raw_input('Continue?(y/n)').lower()

    env.test_init()
    a = ''
    while a != 'n':
        env.restart_test()
        a = raw_input('Continue?(y/n)').lower()
    '''