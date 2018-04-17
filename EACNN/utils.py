#coding:utf-8
import os
import time
import pickle
import numpy as np
from progress.bar import Bar

class ProgressBar(Bar):
    message = 'Loading'
    fill = '#'
    suffix = '%(percent).1f%% | ETA: %(eta)ds'


class CharEmb:
    """
    Generate bow-like character embedding 
    Total 50 chars: "abcdefghijklmnopqrstuvwxyz0123456789,;!?:._'"-@#$%"
    """
    def __init__(self):
        self.chars = """abcdefghijklmnopqrstuvwxyz0123456789,;!?:._'"-@#$%"""
        self.dim = len(self.chars)
        self.char_dict = {}
        for c in self.chars:
            self.char_dict[c] = len(self.char_dict)

    def char_emb(self, word):
        '''
        Generate character embedding for words
        '''
        char_vec = np.zeros(len(self.chars))
        for c in word.lower():
            if c in self.char_dict.keys():
                char_vec[self.char_dict[c]] += 3 #1
        return char_vec


def ten_fold_split_idx(num_data, fname, random=True, k=10):
    """
    Split data for 10-fold-cross-validation
    Split randomly or sequentially
    Retutn the indecies of splited data
    """
    print('Getting tenfold indices ...')
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            print('Loading tenfold indices from %s\n' % fname)
            indices = pickle.load(f)
            return indices
    n = num_data/k
    indices = []

    if random:
        parts = [n]*(k-1)
        parts.append(num_data - n*(k-1))
        flag = [0]*num_data
        for num in parts:
            tmp_idx = []
            while len(tmp_idx) < num:
                idx = np.random.randint(num_data)
                if flag[idx]:
                    continue
                tmp_idx.append(idx)
                flag[idx] = 1
            indices.append(tmp_idx)
    else:
        for i in xrange(k):
            indices.append(range(i*n, (i+1)*n))

    with open(fname, 'wb') as f:
        pickle.dump(indices, f)
    return indices


def index2data(indices, data):
    """
    Split data according to given indices
    """
    print('Spliting data according to indices ...')
    folds = {'train': [], 'valid': []}
    num_data = len(data['texts'])
    for i in xrange(len(indices)):
        valid_texts = []
        valid_tags = []
        train_texts = []
        train_tags = []
        for idx in xrange(num_data):
            if idx in indices[i]:
                valid_texts.append(data['texts'][idx])
                valid_tags.append(data['tags'][idx])
            else:
                train_texts.append(data['texts'][idx])
                train_tags.append(data['tags'][idx])

        valid_data = {'texts': valid_texts, 'tags': valid_tags}
        train_data = {'texts': train_texts, 'tags': train_tags}
        folds['train'].append(train_data)
        folds['valid'].append(valid_data)

    return folds