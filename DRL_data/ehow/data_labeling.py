import re
import os
import ipdb
import time
import pickle


source = 'selected_data/'
save_file = 'ehow_data_1_2k.pkl'
save_labeled_data = 'labeled_ehow_data.pkl'

def split_sents():
    num = 1
    ipdb.set_trace()
    texts = []
    for fname in os.listdir(source):
        with open(source + fname) as f:
            text = f.read()
            atext = re.sub(r'\n|,|;', ' ', text)
            btext = re.split(r'\. |\? |\! ', atext)
            texts.append(btext[:-1])
            with open('out_data/%d.txt' % num, 'w') as f1:
                print num
                f1.write('\n'.join(btext))
                num += 1

    with open(save_file, 'w') as outfile:
        pickle.dump(texts, outfile)


def revise_data():
    with open(save_labeled_data, 'rb') as f:
        print 'Load data from %s...\n' % save_labeled_data
        start_text, start_sent, data = pickle.load(f)
        print 'start_text: %d\t start_sent: %d\n' % (start_text, start_sent)
        num = raw_input('Input a sent num to remove\n')
        num = int(num)
        data[start_text][num] = {}
        start_sent -= 1
        ipdb.set_trace()

    with open(save_labeled_data, 'wb') as f1:
        pickle.dump([start_text, num - 1, data], f1)



def text_labeling():
    with open(save_file, 'r') as f:
        texts = pickle.load(f)
    
    if os.path.exists(save_labeled_data):
        with open(save_labeled_data, 'rb') as f:
            print 'Load data from %s...\n' % save_labeled_data
            last_text, last_sent, data = pickle.load(f)
            print 'last_text: %d\t last_sent: %d\n' % (last_text, last_sent)
        while True:
            init = raw_input('Input last text num and sent num\n')
            if not init:
                print 'No input, program exit!\n'
            if len(init.split()) == 2:
                start_text = int(init.split()[0])
                start_sent = int(init.split()[1])
                break
        ipdb.set_trace()
    else:
        start_text = start_sent = 0
        data = [[] for _ in xrange(64)]
    
    for i in xrange(start_text, 64):
        text = texts[i]
        sents_num = len(text)
        if i == start_text and len(data[i]) > 0:
            sents = data[i]
        else:
            sents = [{} for _ in xrange(sents_num)]
        try:
            if i != start_text:
                start_sent = 0       
            for j in xrange(start_sent, sents_num):
                sent = {}
                this_sent = text[j].split() 
                if j > 0: # print two sentences, used for coreference resolution
                    last_sent = text[j - 1].split()
                else:
                    last_sent = []
                sent['last_sent'] = last_sent
                sent['this_sent'] = this_sent
                sent['acts'] = {}
                words = last_sent + this_sent
                words_num = len(words)
                print 'T%d of 64, S%d of %d:' % (i, j, sents_num)
                for l,w in enumerate(words):
                    print w+'('+str(l)+')',
                while True:
                    act = raw_input('\nInput an action index:\n')
                    if not act:
                        break
                    if len(act.split()) > 1:
                        continue
                    if act == 'r':
                        return    
                    act_idx = int(act)
                    if act_idx >= words_num:
                        print 'action index %d out of range' % act_idx
                        continue
                    objs = raw_input('Input object indices:\n')
                    obj_idx = []
                    continue_flag = False
                    for o in objs.split():
                        idx = int(o)
                        if idx >= words_num:
                            print 'object index %d out of range' % idx
                            continue_flag = True
                            break
                        obj_idx.append(idx)
                    if continue_flag:
                        continue
                    obj_names = [words[k] for k in obj_idx]
                    print '\t%s(%s)' % (words[act_idx], ','.join(obj_names))
                    sent['acts'][act_idx] = obj_idx
                sents[j] = sent
        except Exception as e:
            print 'Error:',e
            data[i] = sents
            with open(save_labeled_data, 'wb') as f:
                pickle.dump([i, j, data], f)
                break_flag = True
                print 'last_text: %d\t last_sent: %d\n' % (i, j)
                break
        data[i] = sents
           
        


if __name__ == '__main__':
    #split_sents()
    break_flag = False
    #while True:
    text_labeling()
    #revise_data()