import re
import os
import pickle
import mysql.connector
import numpy as np
from tqdm import tqdm
#from gensim.models.word2vec import Word2Vec

def pre_process():
    with open('./data/databag3.pickle','r') as f:
        raw_data = pickle.load(f)
    with open('./data/valselect.pickle', 'r') as f:
        devset = pickle.load(f) # selected validation data
    names_map = ['grid', 'jelly', 'l']
    sents = {
        'train': {},
        'dev': {}
    }
    for name_map in names_map:
        sents['train'][name_map] = []
        sents['dev'][name_map] = []
        for idx_data, data in enumerate(raw_data[name_map]):
            instruction = data['instruction']
            if idx_data in devset[name_map]: # find the selected validation
                sents['dev'][name_map].append(instruction)
            else:
                sents['train'][name_map].append(instruction)

    all_data = raw_data['all']
    vocab = {}
    for i in tqdm(range(len(all_data))):
        instruction = all_data[i]['instruction']
        for w in instruction:
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1

    with open('./data/sail_vocab.pkl', 'w') as f:
        pickle.dump(vocab, f)
    with open('./data/sail_sents.pkl','w') as f:
        pickle.dump(sents, f)
    print 'Successfully save pickle file.\n'

def pre_process_database(table, databag_name):
    db = mysql.connector.connect(user='fengwf', password='123', database='test')
    cur = db.cursor()
    if table == 'tag_actions2':
        print "getting long text of WHS ..."
        get_data = 'select * from %s where text_num >= 33'%table
    else:
        get_data = 'select * from %s'%table
    cur.execute(get_data)
    result = cur.fetchall()
    print 'Total sentence: %d\n'%len(result)

    sents = {
        'instruction': [],
        'action':[]
    }
    actions = []
    vocab = {}
    word2ind = {}
    ind2word = {}
    index = 1 # index start from 1, 0 reserved for EOF flag
    for idx,data in enumerate(result):
        words = re.split(r' ', data[2]) #data[2].split() #
        tags = [int(t) for t in re.split(r' ', data[3])] #data[3].split()] #
        if len(words) != len(tags):
            print 'idx %d not match, len(words)=%d, len(tags)=%d'%(idx, len(words), len(tags))
            print data
            import ipdb 
            ipdb.set_trace()
            continue
        #assert len(words) == len(tags)
        sents['instruction'].append(words)
        #temp_action = []
        for i,w in enumerate(words):
            #if tags[i] == '1':
            #    temp_action.append(words[i])
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1
            if w not in word2ind.keys():
                word2ind[w] = index
                ind2word[index] = w
                index += 1
        sents['action'].append(tags)
        #sents['action'].append(temp_action)
        # set low frequency word to -1
        word_count = 5
        for w in vocab.keys():
            if vocab[w] < 5:
                word2ind.pop(w)

    databag = {
        'vocab': vocab,
        'sents': sents,
        'word2ind':word2ind,
        'ind2word':ind2word
    }
    with open('./data/%s_databag.pkl'%databag_name, 'w') as f:
        pickle.dump(databag, f)
    #ipdb.set_trace()
    print 'Successfully save file as %s_databag.pkl\n'%databag_name


def k_flod_split(databag_name, k=10):
    import random
    #import ipdb
    #ipdb.set_trace()
    with open('./data/%s_databag.pkl'%databag_name, 'r') as f:
        raw_data = pickle.load(f)
        dict_data = raw_data['sents']
        num_samples = len(dict_data['instruction'])
    num_slice = int(num_samples/float(k))
    element_list = range(num_samples)
    k_slice_data = []
    left_num = num_samples
    while left_num >= 2*num_slice:
        a_slice = random.sample(element_list, num_slice)
        for i in xrange(num_slice):
            #print i,a_slice[i]
            element_list.remove(a_slice[i])
        left_num = len(element_list)
        k_slice_data.append(a_slice)
    k_slice_data.append(element_list)
    assert len(k_slice_data) == k
    with open('./data/%s_k_slice_data.pkl'%databag_name, 'w') as f:
        pickle.dump(k_slice_data, f)
    print 'Successfully save k slices of data.\n'


class DataProcess(object):
    """docstring for DataProcess"""
    def __init__(self, data_fold, databag, path_rawdata=None):
        print 'initializing the processer ...'
        self.databag = databag
        if path_rawdata:
            self.path_rawdata = path_rawdata
        else:
            self.path_rawdata = './data/'

        # vocabulary of navigation instructions
        # raw_data is a dict
        with open(self.path_rawdata+'%s_databag.pkl'%self.databag, 'r') as f:
            raw_data = pickle.load(f)
            self.vocab = raw_data['vocab']
            self.dict_data = raw_data['sents']
            self.word2ind = raw_data['word2ind']
            self.ind2word = raw_data['ind2word']

        with open(self.path_rawdata+'%s_k_slice_data.pkl'%self.databag, 'r') as f:
            self.k_slice_data = pickle.load(f)
            self.devset = self.k_slice_data[data_fold]
            print 'loading %d slices of data, validate the %dth slice\n'%(len(self.k_slice_data), data_fold)
    
        self.dim_action = 68 #50 #len(self.vocab) + 1
        self.dim_lang = len(self.vocab) + 1
        self.dim_model = 100  # wordvec size
        print 'dim_action: %d\tdim_model: %d\n'%(self.dim_action, self.dim_model)

        self.seq_lang_numpy = None  # instruction to word dict, it's a vector
        self.seq_action_numpy = None  # action to index
        #model_dir = '/home/fengwf/Documents/mymodel5-5-100'
        #self.model = Word2Vec.load_word2vec_format(model_dir, binary=False)


    def process_one_data(self, idx_data):
        # tag_split ='train' or 'dev'
        # one_data means an instruction of a map        

        self.seq_lang_numpy = []
        self.seq_action_numpy = [] # the label of actions, 0 for the EOF flag
        self.action_word = []

        for w in self.dict_data['instruction'][idx_data]:
            self.seq_lang_numpy.append(self.word2ind[w])
        for i,t in enumerate(self.dict_data['action'][idx_data]):
            if t == 1:
                #self.seq_action_numpy.append(self.seq_lang_numpy[i])
                self.seq_action_numpy.append(i)
                self.action_word.append(self.dict_data['instruction'][idx_data][i])
        #self.seq_action_numpy.append(-1)

        self.seq_lang_numpy = np.array(self.seq_lang_numpy, dtype=np.int32)
        self.seq_action_numpy = np.array(self.seq_action_numpy, dtype=np.int32) 
        '''
        for w in one_data:
            if len(self.model[w]):
                word_vec = self.model[w]
            else:
                word_vec = np.zeros(self.dim_model)
            self.seq_lang_numpy.append(word_vec)
        '''



if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    #tables = ['tag_actions', 'tag_actions1', 'tag_actions2', 'tag_actions3', 'tag_actions5']
    #databag_name = ['cooking', 'wikihow', 'windows', 'sail', 'long_wikihow']
    #for i in xrange(len(tables)):
    #    pre_process_database(tables[i], databag_name[i])
    #    k_flod_split(databag_name[i])

    pre_process_database('tag_actions2', 'long_windows')
    k_flod_split('long_windows')
