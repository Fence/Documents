#coding:utf-8
import re
import ipdb
import time
import pickle
import mysql.connector
import numpy as np
from utils import ten_fold_split_idx, index2data

#create table tag_actions5(text_num int, sent_num int, sent varchar(400), tag_sent varchar(400));
class AFEnvironment:
    def __init__(self, args):
        print('Initializing the AFEnvironment...')  
        self.context_len = args.context_len
        self.words_num = self.context_len * 2 + 1
        self.emb_dim = args.emb_dim
        self.word_dim = args.word_dim
        self.tag_dim = args.tag_dim
        self.object_rate = args.object_rate
        self.object_label = args.action_label
        self.non_object_label = args.non_action_label
        self.reward_base = args.reward_assign
        self.reward_assign = self.reward_base * np.array([2, 1, -1, -2])
        self.distance = np.abs(np.arange(-self.context_len, self.context_len + 1))
        self.terminal_flag = False
        self.batch_act_num = args.batch_act_num
        

    def restart(self, act_idx, current_text, init=False):
        #ipdb.set_trace()
        if init:
            self.terminal_flag = False
            context = np.zeros([self.words_num, self.word_dim])
            tags = np.ones([self.words_num, self.tag_dim])
            for i in range(act_idx - self.context_len, act_idx + self.context_len + 1):
                if 0 <= i < self.word_dim:
                    context[i + self.context_len] = current_text['sent_vec'][i]
            if act_idx in current_text['acts']: # real action
                for obj_idx in current_text['acts'][act_idx]:
                    if obj_idx == 'NULL':
                        continue
                    if 0 <= obj_idx - act_idx + self.context_len < self.words_num:
                        tags[obj_idx - act_idx + self.context_len] = 2 #self.object_label
            self.text_vec = np.concatenate((context, tags), axis=1)
        self.state = self.text_vec.copy() # NB!
        self.state[:,self.word_dim:] = 0

        
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
            if act_str[i] == '1':  
                self.state[word_idx,self.word_dim:] = self.object_label  
            else: # act_str[i] == '1'
                self.state[word_idx,self.word_dim:] = self.non_object_label    
            t_a_count = 0  #amount of tagged actions 
            for t in self.state[:,-1]:
                if t == self.object_label:
                    t_a_count += 1
            t_a_rate = float(t_a_count)/self.words_num

            label = self.text_vec[word_idx,-1]
            # text_vec is labelled data
            if label == self.state[word_idx,-1]:
                if label == self.object_label:
                    reward = self.reward_assign[0]
                else:
                    reward = self.reward_assign[1] 
            else:
                if self.text_vec[word_idx,-1] == self.non_object_label:
                    reward = self.reward_assign[2]
                else:
                    reward = self.reward_assign[3]
            if t_a_rate <= self.object_rate:
                reward += 5.0 * np.square(t_a_rate) * self.reward_base
            else:
                reward -= 5.0 * np.square(t_a_rate) * self.reward_base
            bacth_act_reward += reward
        # all words of current text are tagged, break
        if word_idx >= self.words_num - 1:
            self.terminal_flag = True

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
    parser.add_argument("--object_rate", type=float, default=0.15, help="")
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