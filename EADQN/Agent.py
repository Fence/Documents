#coding:utf-8
import ipdb
import time
import random
import numpy as np
from tqdm import tqdm

class Agent:
    def __init__(self, environment, replay_memory, deep_q_network, args):
        print('Initializing the Agent...')
        self.env = environment
        self.mem = replay_memory
        self.net = deep_q_network
        self.num_actions = args.num_actions
        self.words_num = args.words_num
        self.batch_size = args.batch_size

        self.exp_rate_start = args.exploration_rate_start
        self.exp_rate_end = args.exploration_rate_end
        self.exp_decay_steps = args.exploration_decay_steps
        self.exploration_rate_test = args.exploration_rate_test
        self.total_train_steps = args.start_epoch * args.train_steps

        self.train_frequency = args.train_frequency
        self.train_repeat = args.train_repeat
        self.target_steps = args.target_steps
        
        self.steps = 0  #use to decrease the reward during time steps 
        self.action_label = args.action_label
        self.random_play = args.random_play
        self.batch_act_num = args.batch_act_num
        self.epoch_end_flag = False

    
    def _restart(self, train_flag, init=False):
        #print('\nRestarting in agent, train_flag = {}, init = {}'.format(train_flag, init))
        self.steps = 0
        self.env.restart(train_flag, init)
        self.epoch_end_flag = self.env.epoch_end_flag


    def _explorationRate(self):
        # calculate decaying exploration rate
        if self.total_train_steps < self.exp_decay_steps:
            return self.exp_rate_start - self.total_train_steps * \
            (self.exp_rate_start - self.exp_rate_end) / self.exp_decay_steps
        else:
            return self.exp_rate_end
 

    def step(self, exploration_rate, train_flag=False):
        # exploration rate determines the probability of random moves
        if random.random() < exploration_rate:
            action = np.random.randint(2**self.batch_act_num)
        else:
            # otherwise choose action with highest Q-value
            current_state = self.env.getState()
            qvalues = self.net.predict(current_state)
            action = np.argmax(qvalues[0])
            assert len(qvalues[0]) == 2**self.batch_act_num
            
        # perform the action  
        reward = self.env.act(action, self.steps)
        state = self.env.getState()
        terminal = self.env.isTerminal()
        
        self.steps += 1
        if not terminal:
            #decrease the reward with time steps
            #reward -= abs(reward)*self.steps/(1.5*self.num_actions)
            results = []  
        else:
            results = self.compute_f1()
            self.steps = 0
            #reward += 2   #give a bonus to the terminal actions
            self._restart(train_flag)

        return action, reward, state, terminal, results

 
    def train(self, train_steps, epoch = 0):
        '''
        Play given number of steps
        '''
        #ipdb.set_trace()
        self._restart(train_flag=True, init=True)
        for i in xrange(train_steps):
            if i%100 == 0:
                print('\n\nepoch: %d  Training step: %d' % (epoch,i))

            if self.random_play:
                action, reward, state, terminal, _ = self.step(1, train_flag=True)
            else:
                action, reward, state, terminal, _ = self.step(self._explorationRate(), train_flag=True)
                self.mem.add(action, reward, state, terminal)

                # Update target network every target_steps steps
                if self.target_steps and i % self.target_steps == 0:
                    self.net.update_target_network()

                # train after every train_frequency steps
                if self.mem.count > self.mem.batch_size and i % self.train_frequency == 0:
                    # train for train_repeat times
                    for j in xrange(self.train_repeat):
                        # sample minibatch
                        minibatch = self.mem.getMinibatch()
                        # train the network
                        self.net.train(minibatch)
            
            # increase number of training steps for epsilon decay
            self.total_train_steps += 1
            if self.epoch_end_flag:
                break
    

    def test(self, test_steps, f1):
        '''
        Play given number of steps
        '''
        t_right_tag = t_right_acts = t_tagged_acts = t_total_acts = t_words = 0
        t_acc = t_rec = t_pre = t_f1_value = 0.0
        
        self._restart(train_flag=False, init=True)
        for i in tqdm(xrange(test_steps)):
            if self.random_play:
                a, r, s, t, rs = self.step(1, train_flag=False)
            else:
                a, r, s, t, rs = self.step(self.exploration_rate_test, train_flag=False)
            if t:
                temp_words, total_acts, tagged_acts, right_acts, right_tag, acc, rec, pre, f1_value = rs
                f1.write('\nText: %d\ntotal words: %d\n' % (self.env.valid_text_idx - 1, temp_words))
                f1.write('total: %d\tright: %d\ttagged: %d\n' % (total_acts, right_acts, tagged_acts))  
                f1.write('acc: %f\trec: %f\tpre: %f\tf1: %f\n' % (acc, rec, pre, f1_value))

                t_words += temp_words
                t_right_tag += right_tag                  
                t_right_acts += right_acts
                t_tagged_acts += tagged_acts
                t_total_acts += total_acts    

            if self.epoch_end_flag:
                break   

        t_acc = float(t_right_tag)/t_words
        if t_total_acts > 0:
            t_rec = float(t_right_acts)/t_total_acts
        if t_tagged_acts > 0:
            t_pre = float(t_right_acts)/t_tagged_acts
        if t_rec + t_pre > 0:
            t_f1_value = (2.0 * t_rec * t_pre)/(t_rec + t_pre)

        f1.write('\n\nSummary:\n')
        f1.write('total: %d\tright: %d\ttagged: %d\n' % (t_total_acts, t_right_acts, t_tagged_acts))  
        f1.write('acc: %f\trec: %f\tpre: %f\tf1: %f\n' % (t_acc, t_rec, t_pre, t_f1_value))
        print('\n\nSummary:')
        print('total: %d\tright: %d\ttagged: %d' % (t_total_acts, t_right_acts, t_tagged_acts))  
        print('acc: %f\trec: %f\tpre: %f\tf1: %f\n' % (t_acc, t_rec, t_pre, t_f1_value))

        return t_rec, t_pre, t_f1_value


    def compute_f1(self):
        """
        Compute f1 score for current text
        """
        text_vec_tags = self.env.text_vec[:,-1]
        state_tags = self.env.state[:,-1]
        right_tag = right_acts = tagged_acts = total_acts = 0
        acc = rec = pre = f1_value = 0.0
        
        total_words = self.words_num
        temp_words = len(self.env.current_text['tokens'])
        if temp_words > total_words:
            temp_words = total_words
        for t in text_vec_tags:
            if t == self.action_label:
                total_acts += 1

        for s in xrange(temp_words):
            if state_tags[s] == self.action_label:
                tagged_acts += 1
                if text_vec_tags[s] == state_tags[s]:
                    right_acts += 1
            if text_vec_tags[s] == state_tags[s]:
                right_tag += 1

        acc = float(right_tag)/temp_words
        if total_acts > 0:
            rec = float(right_acts)/total_acts
        if tagged_acts > 0:
            pre = float(right_acts)/tagged_acts
        if rec + pre > 0:
            f1_value = (2.0 * rec * pre)/(rec + pre)

        print('\ntotal words: %d\n' % temp_words)
        print('total: %d\tright: %d\ttagged: %d' % (total_acts, right_acts, tagged_acts))  
        print('acc: %f\trec: %f\tpre: %f\tf1: %f\n' % (acc, rec, pre, f1_value))

        return temp_words, total_acts, tagged_acts, right_acts, right_tag, acc, rec, pre, f1_value