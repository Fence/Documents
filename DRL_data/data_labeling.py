import re
import os
import sys
import ipdb
import time
import json
import pickle
from tqdm import tqdm


class QuitProgram(Exception):
    def __init__(self, message='Quit the program.\n'):
        Exception.__init__(self)
        self.message = message

class TextParsing(object):
    """docstring for TextParsing"""
    def __init__(self):
        from nltk.stem import WordNetLemmatizer
        from nltk.parse.stanford import StanfordDependencyParser
        core = '/home/fengwf/stanford/stanford-corenlp-3.7.0.jar'
        model = '/home/fengwf/stanford/english-models.jar'
        self.dep_parser = StanfordDependencyParser(path_to_jar=core, path_to_models_jar=model,
                        encoding='utf8', java_options='-mx2000m')
        self.lemma = WordNetLemmatizer()
            

    def build_vocab(self, save_name):
        # e.g. save_name = 'wikihow/wikihow_act_seq.pkl'
        with open(save_name, 'rb') as f:
            data = pickle.load(f)

        word_dict = {}
        verb_dict = {}
        objs_dict = {}
        for text in data:
            for act, objs in text['act_seq']:
                act = act.lower()
                if act not in word_dict:
                    word_dict[act] = 1
                else:
                    word_dict[act] += 1
                if act not in verb_dict:
                    verb_dict[act] = 1
                else:
                    verb_dict[act] += 1
                for obj in objs.split('_'):
                    if obj not in word_dict:
                        word_dict[obj] = 1
                    else:
                        word_dict[obj] += 1
                    if obj not in objs_dict:
                        objs_dict[obj] = 1
                    else:
                        objs_dict[obj] += 1
        ipdb.set_trace()
        words = sorted(word_dict.items(), key=lambda x:x[1], reverse=True)
        verbs = sorted(verb_dict.items(), key=lambda x:x[1], reverse=True)
        objs = sorted(objs_dict.items(), key=lambda x:x[1], reverse=True)
        print(len(word_dict), len(verb_dict), len(objs_dict))


    def stanford_find_vp_details(self, indata_name, outdata_name):

        data = []
        with open(indata_name, 'rb') as f0:
            indata = pickle.load(f0)[-1]
        if os.path.exists('%s.pkl' % outdata_name):
            print('Loading data...')
            data = pickle.load(open('%s.pkl' % outdata_name, 'rb'))
            print('len(data) = %d' % len(data))

        try:
            count = 0
            for cate in indata:
                print(cate)
                for page in indata[cate]:
                    if 'detail' not in page:
                        continue
                    for detail in page['detail']:
                        count += 1
                        if count <= len(data):
                            continue
                        tmp_data = {'title': page['title']}
                        tmp_data['sent'] = []
                        tmp_data['act_seq'] = []
                        tmp_data['dep_conll'] = []

                        sents = []
                        for step in detail:
                            text = re.sub(r'\[.*\]|/', '', step)
                            text = re.sub(r'[\n\r]', ' ', text)
                            tmp_sents = re.split(r'\. |\? |\! ', text)
                            for s in tmp_sents:
                                if len(s.strip().split()) > 1:
                                    sents.append(s.strip())
                        try:
                            dep = self.dep_parser.raw_parse_sents(sents) 
                        except AssertionError:
                            print('Raise AssertionError')
                            sents = [' '.join(re.findall(r'[\w\'\.]+', s)) for s in sents]
                            try:
                                dep = self.dep_parser.raw_parse_sents(sents) 
                            except Exception as e:
                                print(e)
                                continue
                        except Exception as e:
                            print(e)
                            continue

                        for j in range(len(sents)):
                            try:
                                dep_root = next(dep)
                                dep_sent = next(dep_root)
                            except StopIteration:
                                print('j = %d len(sents) = %d Raise StopIteration.\n' % (j, len(sents)))
                                break
                            conll = [_.split() for _ in str(dep_sent.to_conll(10)).split('\n') if _]
                            words = []
                            idx2word = {}
                            for w in conll:
                                idx2word[w[0]] = w[1]
                                words.append(w[1])
                            tmp_data['sent'].append(' '.join(words))
                            #tmp_data['dep_conll'].append(conll)
                            for line in conll:
                                if 'dobj' in line or 'nsubjpass' in line:
                                    obj = [line[1]]
                                    obj_idxs = [line[0]]
                                    verb_idx = line[6]
                                    for one_line in conll:
                                        if one_line[6] == obj_idxs[0] and one_line[7] == 'conj':
                                            obj.append(one_line[1])
                                            obj_idxs.append(one_line[0])
                                    act = idx2word[verb_idx].lower()
                                    act_obj_pair = (act, '_'.join(obj))
                                    tmp_data['act_seq'].append(act_obj_pair)

                        data.append(tmp_data)
                        print(len(data), page['title'])
                        if len(data) % 2000 == 0:
                            print('len(data): %d, try to save file.' % len(data))
                            self.save_txt_and_pkl(outdata_name, data)
                            print('Successfully save %s\n' % outdata_name)
                        elif len(data) % 1000 == 0:
                            print('len(data): %d, try to save file.' % len(data))
                            self.save_txt_and_pkl(outdata_name, data, False)
                            print('Successfully save %s\n' % outdata_name)
        except KeyboardInterrupt:
            print('Manually keyboard interrupt!\n')
        except Exception as e:
            print(e)
        print('len(data): %d, try to save file.' % len(data))
        self.save_txt_and_pkl(outdata_name, data)
        print('Successfully save %s\n' % outdata_name)


    def stanford_find_vp(self, indata_name, outdata_name):

        num_texts = 124 # 96 #
        source = 'wikihow/new_details/' #'ehow/out_data/' # 'cooking/out_data/' #
        #save_name = 'wikihow/wikihow_act_seq_100k' # 'ehow/ehow_act_seq' # 'cooking/cooking_act_seq' #
        data = []
        #ipdb.set_trace()
        with open(indata_name, 'rb') as f0:
            indata = pickle.load(f0)[-1]
        if os.path.exists('%s.pkl' % outdata_name):
            print('Loading data...')
            data = pickle.load(open('%s.pkl' % outdata_name, 'rb'))
            print('len(data) = %d' % len(data))
        #for i in range(num_texts):
        #for name in os.listdir(source):
        try:
            count = 0
            for cate in indata:
                print(cate)
                for page in indata[cate]:
                    if 'sub_task' not in page:
                        continue
                    for sub_task in page['sub_task']:
                        count += 1
                        if count <= len(data):
                            continue
                        text = '\n'.join(sub_task)
                        tmp_data = {'title': page['title']}
                        tmp_data['sent'] = []
                        tmp_data['act_seq'] = []
                        tmp_data['dep_conll'] = []
                        #fname = '%s%d.txt' % (source, i + 1)
                        #fname = source + name
                        #print(fname)
                        #try:
                        #text = open(fname).read()
                        text = re.sub(r'/', ' ', text)
                        sents = text.split('\n') #.readlines()
                        try:
                            dep = self.dep_parser.raw_parse_sents(sents) 
                        except AssertionError:
                            #print('\n', sents)
                            print('Raise AssertionError')
                            sents = [' '.join(re.findall(r'[\w\'\.]+', s)) for s in sents]
                            #print(sents, '\n')
                            try:
                                dep = self.dep_parser.raw_parse_sents(sents) 
                            except Exception as e:
                                print(e)
                                continue
                        except Exception as e:
                            print(e)
                            continue
         
                        for j in range(len(sents)):
                            try:
                                dep_root = next(dep)
                                dep_sent = next(dep_root)
                            except StopIteration:
                                print('j = %d len(sents) = %d Raise StopIteration.\n' % (j, len(sents)))
                                break
                            conll = [_.split() for _ in str(dep_sent.to_conll(10)).split('\n') if _]
                            words = []
                            idx2word = {}
                            for w in conll:
                                idx2word[w[0]] = w[1]
                                #word_lemma = self.lemma.lemmatize(w[1])
                                #if word_lemma == 'pythonly':
                                #    word_lemma = w[1]
                                words.append(w[1]) #word_lemma
                            tmp_data['sent'].append(' '.join(words))
                            tmp_data['dep_conll'].append(conll)
                            for line in conll:
                                if 'dobj' in line or 'nsubjpass' in line:
                                    obj = [line[1]]
                                    obj_idxs = [line[0]]
                                    verb_idx = line[6]
                                    for one_line in conll:
                                        if one_line[6] == obj_idxs[0] and one_line[7] == 'conj':
                                            obj.append(one_line[1])
                                            obj_idxs.append(one_line[0])
                                    # lemmatize, find the original word of action
                                    act = idx2word[verb_idx].lower()
                                    act_obj_pair = (act, '_'.join(obj))
                                    tmp_data['act_seq'].append(act_obj_pair)

                        data.append(tmp_data)
                        print(len(data), page['title'])
                        if len(data) % 2000 == 0:
                            print('len(data): %d, try to save file.' % len(data))
                            self.save_txt_and_pkl(outdata_name, data)
                            print('Successfully save %s\n' % outdata_name)
                        elif len(data) % 1000 == 0:
                            print('len(data): %d, try to save file.' % len(data))
                            self.save_txt_and_pkl(outdata_name, data, False)
                            print('Successfully save %s\n' % outdata_name)
        except KeyboardInterrupt:
            print('Manually keyboard interrupt!\n')
        except Exception as e:
            print(e)
        print('len(data): %d, try to save file.' % len(data))    
        self.save_txt_and_pkl(outdata_name, data)
        print('Successfully save %s' % outdata_name)
        
                        
    def get_labeled_win2k(self):
        text = open('win2k/window2k_annotations.txt').read()
        articles = text.split('-------------------------------------------------')[1:]

        data = []
        #ipdb.set_trace()
        for article in articles:
            tmp_data = {}
            tmp_data['sent'] = []
            tmp_data['act_seq'] = []
            lines = article.split('\n')
            for line in lines:
                if line.startswith('article', 4):
                    print(line.strip())
                elif line.startswith('c:', 9):
                    pass
                elif line.startswith('- ', 9) or line.startswith('~ ', 9):
                    a = re.split(r'\([\w\-\/\:]*\)', line)
                    assert len(a) >= 2
                    act = re.findall(r'\w+', a[0])
                    obj = re.findall(r'\w+', a[1])
                    if len(act) == 0 or len(obj) == 0:
                        ipdb.set_trace()
                    act = self.lemma.lemmatize('_'.join(act).lower(), pos='v')
                    obj = '_'.join(obj).lower()
                    act_obj_pair = (act, obj)
                    tmp_data['act_seq'].append(act_obj_pair)
                elif len(line.strip()):
                    tmp_data['sent'].append(line.strip())
            if len(tmp_data['act_seq']) == 0:
                ipdb.set_trace()
            data.append(tmp_data)
        self.save_txt_and_pkl('win2k/win2k_act_seq', data, True, protocol=2)


    def save_txt_and_pkl(self, fname, data, save_txt=False, protocol=3):
        with open('%s.pkl'%fname, 'wb') as f1:
            pickle.dump(data, f1, protocol=protocol)
        if save_txt:
            with open('%s.txt'%fname, 'w') as f0:
                count = 0
                for d in data:
                    count += 1
                    if 'title' in d:
                        try:
                            f0.write('<Article %d>: %s\n' % (count, d['title']))
                        except Exception as e:
                            print('An error occurs in saving file', e)
                    else:
                        try:
                            f0.write('<Article %d>: \n' % count)
                        except Exception as e:
                            print('An error occurs in saving file', e)
                    for i, s in enumerate(d['sent']):
                        try:
                            f0.write('<Sentence %d>: %s\n' % (i, s))
                        except Exception as e:
                            print('An error occurs in saving file', e)
                    f0.write('\n')
                    for j, (act, obj) in enumerate(d['act_seq']):
                        try:
                            f0.write('<Action %d>: %s  %s\n' % (j, act, obj))
                        except Exception as e:
                            print('An error occurs in saving file', e)
                    f0.write('\n')
                    #if 'dep_conll' in d.keys():
                    #    f0.write('\n<Dependency>\n')
                    #    for dc in d['dep_conll']:
                    #        for c in dc:
                    #            f0.write(' '.join(c)+'\n')
                    #        f0.write('\n')
                    f0.write('\n')


class DataLabeler(object):
    """for wikihow dataset: 
    1. find top 500 'or texts' of home and garden
    2. text labeling, add annotations: action types, action indexes, object indexes
    3. add object type, split object indexes by 'or, Or' 
    """
    def __init__(self):
        self.num_texts = 154
        self.one_line_data = 1
        self.home = 'wikihow' #'ehow' #'cooking' #'win2k' #wikihow
        self.source = '%s/raw_data/' % self.home
        self.out_path = '%s/out_data/' % self.home
        self.save_file = '%s/%s_data.pkl' % (self.home, self.home)
        self.save_labeled_data = '%s/labeled_%s_data.pkl' % (self.home, self.home)
        self.refined_data = '%s/refined_%s_data.pkl' % (self.home, self.home)
    

    def find_top_or_text_by_category(self):
        print('Loading data...')
        data = pickle.load(open('wikihow/wikihow_data_100k.pkl', 'rb'))[-1]
        garden = data['Category:Home-and-Garden']
        print(len(garden))
        texts = []
        for page in garden:
            if 'detail' not in page or len(page['detail']) != len(page['task']):
                continue
            for i, detail in enumerate(page['detail']):
                sents = []
                for step in detail:
                    text = re.sub(r'\[.*\]|/', '', step)
                    text = re.sub(r'[\n\r]', ' ', text)
                    tmp_sents = re.split(r'\. |\? |\! ', text)
                    for s in tmp_sents:
                        s = re.sub(r'<.*>|<*', '', s)
                        if len(s.strip().split()) > 1:
                            sents.append(s.strip())
                texts.append({'title': page['task'][i], 'sent': sents})
        with open('wikihow/wikihow_home_and_garden_data.pkl', 'wb') as f:
            pickle.dump(texts, f)
        self.find_top_or_text('', 'wikihow/home_and_garden_500_words', texts)


    def find_top_or_text(self, infile, outfile, texts='', topn=-1):
        # infile = 'wikihow/wikihow_act_seq_152k_details.pkl'
        # outfile = 'top1000_or_texts_152k.txt'
        if texts:
            data = texts
        else:
            data = pickle.load(open(infile, 'rb'))
        or_dicts = {}
        for i, text in enumerate(data):
            #if len(text['sent']) > 40:
            #    continue
            if sum([len(s.split()) for s in text['sent']]) > 500:
                continue
            for sent in text['sent']:
                words = sent.split()
                if 'or' in words or 'Or' in words:
                    if i not in or_dicts:
                        or_dicts[i] = 1
                    else:
                        or_dicts[i] += 1

        or_list = sorted(or_dicts.items(), key=lambda x:x[1], reverse=True)
        f = open(outfile + '.txt', 'w')
        if topn <= 0:
            topn = len(or_list)
        print('topn:', topn)
        texts = []
        for idx in or_list[: topn]:
            text = data[idx[0]]['sent']
            sents = [re.sub(r',|;|:|\.|', '', s) for s in text]
            texts.append(sents)
            f.write('text: %d\n' % idx[0])
            if 'title' in data[idx[0]]:
                f.write('title: %s\n' % data[idx[0]]['title'])
            for j, sent in enumerate(text):
                f.write('No%d: %s\n' % (j, sent))
            f.write('\n\n')
        f.close()
        with open(outfile + '.pkl', 'wb') as f:
            pickle.dump(texts, f, protocol=2)


    def get_sail_data(self):
        lines = open('sail/paragraph.instructions').readlines()
        i = 0
        data = {}
        assert len(lines) % 4 == 0
        for i in range(int(len(lines) / 4)):
            tags = lines[i * 4].split('_')
            key = '_'.join(tags[1: 4])
            #key = lines[i*4]
            sents = lines[i * 4 + 2].replace('\n', ' ')
            sents = sents.split('. ')
            assert len(sents[-1]) == 0 
            words = [s.split() for s in sents[: -1]]
            if key not in data:
                data[key] = [words]
            else:
                data[key].append(words)
        print(len(data))
        total = 0
        for key, sents in data.items():
            words_num = 0
            for sent in sents:
                words_num += sum([len(s) for s in sent])
            total += words_num
            print('{:<20}\t{:<5}\t{:<5}\t{:<5}'.format(key, len(sents), words_num, total))
        ipdb.set_trace()
        for i in sorted(data.items(),key=lambda x:x[1], reverse=True): print(i)


    def add_action_type(self):
        self.save_labeled_data = 'cooking/new_refined_cooking_data2.pkl'
        self.refined_data = 'cooking/new_cooking_labeled_data2.pkl'
        with open(self.save_labeled_data, 'rb') as f:
            data = pickle.load(f)

        last_sent = last_text = 0
        out_data = []
        if os.path.exists(self.refined_data):
            print('Load data from %s...\n' % self.refined_data)
            last_text, last_sent, out_data = pickle.load(open(self.refined_data, 'rb'))
            print('last_text: %d\t last_sent: %d\n' % (last_text, last_sent))
            while True:
                init = input('Input last text num and sent num\n')
                if not init:
                    print('No input, program exit!\n')
                if len(init.split()) == 2:
                    start_text = int(init.split()[0])
                    start_sent = int(init.split()[1])
                    break
            ipdb.set_trace()
        else:
            start_text = start_sent = 0
            out_data = [[] for _ in range(len(data))]
        try:
            for i in range(start_text, len(data)):
                if i == start_text and len(out_data[i]) > 0:
                    out_sents = out_data[i]
                else:
                    out_sents = [{} for _ in range(len(data[i]))]
                if i != start_text:
                    start_sent = 0
                for j in range(start_sent, len(data[i])):
                    print('\nT%d of %d, S%d of %d:' % (i, len(data), j, len(data[i])))
                    sent = data[i][j]
                    words = sent['last_sent'] + sent['this_sent']
                    acts = []
                    for l, w in enumerate(words):
                        print('%s(%d)'%(w, l), end=' ')
                    print('\n')
                    tips = False
                    for w in ['or', 'Or', 'if', 'If', "don't", 'not', 'avoid']:
                        if w in sent['this_sent']:
                            tips = True
                            print("\n%s in words[%d]\n" % (w, words.index(w)))

                    tmp_acts = sorted(sent['acts'].items(), key=lambda x:x[0])
                    for act_idx, obj_idxs in tmp_acts:
                        objs = []
                        for o in obj_idxs:
                            if o >= 0:
                                objs.append(words[o])
                            else:
                                objs.append('NULL')
                        print('%s(%s)'%(words[act_idx], ','.join(objs)))

                    for act_idx, obj_idxs in tmp_acts:
                        print(act_idx, obj_idxs)
                        if not tips:
                            act_type = 1
                            related_acts = []
                            acts.append({'act_idx': act_idx, 'obj_idxs': obj_idxs,
                                            'act_type': act_type, 'related_acts': related_acts})
                            continue
                        while True:
                            inputs = input('\nInput action type and related action indecies:\n')
                            if not inputs:
                                continue
                            if inputs == 'q':
                                last_sent = j
                                last_text = i
                                out_data[i] = out_sents
                                raise QuitProgram()
                            elif inputs == 'r': # revise a sent
                                print(' '.join(sent['this_sent']))
                                text = input('Input right this sentence\n')
                                sent['this_sent'] = text.strip().split()
                                words = sent['last_sent'] + sent['this_sent']
                                for l, w in enumerate(words):
                                    print('%s(%d)'%(w, l), end=' ')
                                continue
                            elif inputs == 'w':
                                print(' '.join(sent['last_sent']))
                                text = input('Input right last sentence\n')
                                sent['last_sent'] = text.strip().split()
                                words = sent['last_sent'] + sent['this_sent']
                                for l, w in enumerate(words):
                                    print('%s(%d)'%(w, l), end=' ')
                                continue
                            elif inputs == 't':
                                #ipdb.set_trace()
                                sent['this_sent'][0] = sent['this_sent'][0][0] + sent['this_sent'][0][1:].lower()
                                for ii in range(1, len(sent['this_sent'])):
                                    sent['this_sent'][ii] = sent['this_sent'][ii].lower()
                                words = sent['last_sent'] + sent['this_sent']
                                for l, w in enumerate(words):
                                    print('%s(%d)'%(w, l), end=' ')
                                continue
                            elif inputs == 'e':
                                sent['last_sent'][0] = sent['last_sent'][0][0] + sent['last_sent'][0][1:].lower()
                                for ii in range(1, len(sent['last_sent'])):
                                    sent['last_sent'][ii] = sent['last_sent'][ii].lower()
                                words = sent['last_sent'] + sent['this_sent']
                                for l, w in enumerate(words):
                                    print('%s(%d)'%(w, l), end=' ')
                                continue
                            inputs = inputs.split()
                            if len(inputs) >= 1:
                                act_type = int(inputs[0])
                                related_acts = [int(ra) for ra in inputs[1: ]]
                                if act_type > 3 or act_type < 1:
                                    print('Wrong action type! act_type should be 1, 2 or 3!')
                                    continue
                                if act_type == 3 and len(related_acts) == 0:
                                    print('Wrong inputs! Missed related actions!')
                                    continue
                                if len(related_acts) > 0:
                                    related_act_words = []
                                    for ra in related_acts:
                                        related_act_words.append(words[ra])
                                    print(act_type, ' '.join(related_act_words))
                                acts.append({'act_idx': act_idx, 'obj_idxs': obj_idxs,
                                            'act_type': act_type, 'related_acts': related_acts})
                                break
                    print(acts)
                    sent['acts'] = acts
                    out_sents[j] = sent
                out_data[i] = out_sents
        except Exception as e:
            print(e)

        with open(self.refined_data, 'wb') as f:
            pickle.dump([i, j, out_data], f, protocol=2)
            print('last_text: %d\t last_sent: %d\n' % (i, j))


    def add_object_type(self):
        with open(self.save_labeled_data, 'rb') as f:
            data = pickle.load(f)[-1]

        last_sent = last_text = 0
        out_data = []
        if os.path.exists(self.refined_data):
            print('Load data from %s...\n' % self.refined_data)
            last_text, last_sent, out_data = pickle.load(open(self.refined_data, 'rb'))
            print('last_text: %d\t last_sent: %d\n' % (last_text, last_sent))
            while True:
                init = input('Input last text num and sent num\n')
                if not init:
                    print('No input, program exit!\n')
                if len(init.split()) == 2:
                    start_text = int(init.split()[0])
                    start_sent = int(init.split()[1])
                    break
            ipdb.set_trace()
        else:
            start_text = start_sent = 0
            out_data = [[] for _ in range(len(data))]
        try:
            for i in range(start_text, len(data)):
                if i == start_text and len(out_data[i]) > 0:
                    if len(out_data[i]) == len(data[i]):
                        out_sents = out_data[i]
                    else:
                        out_sents = [{} for _ in range(len(data[i]))]
                else:
                    out_sents = [{} for _ in range(len(data[i]))] #[]#
                if i != start_text:
                    start_sent = 0
                for j in range(start_sent, len(data[i])):
                    sent = data[i][j]
                    if len(sent) == 0: 
                        #print('\nEmpty sentence: (i=%d, j=%d)\n' % (i, j))
                        continue
                    words = sent['last_sent'] + sent['this_sent']
                    acts = []
                    or_ind = []
                    print_change = False
                    for k, w in enumerate(sent['this_sent']):
                        if w == 'or':
                            or_ind.append(k + len(sent['last_sent']))

                    if len(or_ind) == 0:
                        for act in sent['acts']:
                            act['obj_idxs'] = [act['obj_idxs'], []]
                            acts.append(act)
                    else:
                        for act in sent['acts']:
                            split = None
                            for k in range(len(act['obj_idxs']) - 1):
                                for oi in or_ind:
                                    if act['obj_idxs'][k] < oi < act['obj_idxs'][k+1]:
                                        split = k + 1
                                        break
                                if split != None:
                                    break
                            if split != None:
                                print('\nT%d of %d, S%d of %d:' % (i, len(data), j, len(data[i])))
                                for l, w in enumerate(words):
                                    print('%s(%d)'%(w, l), end=' ')
                                print('\n')
                                print('or_ind: {}\n'.format(or_ind))
                                print('{}({})'.format(act['act_idx'], act['obj_idxs']))
                                confirm = input('Split or not? (y/n)\n')
                                if confirm.lower() == 'y':
                                    act['obj_idxs'] = [act['obj_idxs'][: split], act['obj_idxs'][split: ]]
                                    print('{}({}; {})'.format(act['act_idx'], act['obj_idxs'][0], act['obj_idxs'][1]))
                                    print_change = True
                                else:
                                    act['obj_idxs'] = [act['obj_idxs'], []]
                            else:
                                act['obj_idxs'] = [act['obj_idxs'], []]
                            acts.append(act)
                    if print_change:
                        print('before: {}\n\nafter : {}\n'.format(sent['acts'], acts))
                    sent['acts'] = acts
                    if len(out_sents) < j + 1:
                        out_sents.append({})
                    out_sents[j] = sent
                    #out_sents.append(sent)
                out_data[i] = out_sents
                # if(len(out_sents) != len(data[i])):
                #     #ipdb.set_trace()
                #     time.sleep(1)
                #     print('\nlen(out_sents) != len(data[i]): (i=%d, j=%d)\n' % (i, j))
        except Exception as e:
            ipdb.set_trace()
            print(e)

        with open(self.refined_data, 'wb') as f:
            pickle.dump([i, j, out_data], f, protocol=2)
            print('last_text: %d\t last_sent: %d\n' % (i, j))


    def transfer(self, name):
        _, __, indata = pickle.load(open('%s/refined_%s_data.pkl'%(name, name),'rb'))

        data = []
        tmp_data = {}
        max_sent_len = 0
        max_char_len = 0
        log = {'wrong_last_sent': 0, 'act_reference_1': 0, 'related_act_reference_1': 0,
                'obj_reference_1': 0, 'non-obj_reference_1': 0}
        #ipdb.set_trace()
        for i in range(len(indata)):
            words = []
            sents = []
            word2sent = {}
            text_acts = []
            sent_acts = []
            #if i == 44:
            #    ipdb.set_trace()
            reference_related_acts = False
            for j in range(len(indata[i])):
                if len(indata[i][j]) == 0:
                    print('%s, len(indata[%d][%d]) == 0'%(name, i, j))
                    continue
                last_sent = indata[i][j]['last_sent']
                this_sent = indata[i][j]['this_sent']
                acts = indata[i][j]['acts']
                
                if j > 0 and len(last_sent) != len(indata[i][j-1]['this_sent']):
                    #ipdb.set_trace()
                    b1 = len(last_sent)
                    b2 = len(indata[i][j-1]['this_sent'])
                    for k in range(len(acts)):
                        ai = acts[k]['act_idx']
                        new_act_type = acts[k]['act_type']
                        new_act_idx = ai - b1 + b2
                        new_obj_idxs = [[],[]]
                        for l in range(2):
                            for oi in acts[k]['obj_idxs'][l]:
                                if oi == -1:
                                    new_obj_idxs[l].append(oi)
                                else:
                                    new_obj_idxs[l].append(oi - b1 + b2)
                            assert len(new_obj_idxs[l]) == len(acts[k]['obj_idxs'][l])
                        #if len(acts[k]['related_acts']) > 0:
                        #    ipdb.set_trace()
                        new_related_acts = []
                        acts[k] = {'act_idx': new_act_idx, 'obj_idxs': new_obj_idxs,
                                'act_type': new_act_type, 'related_acts': new_related_acts}
                    last_sent = indata[i][j-1]['this_sent']
                    log['wrong_last_sent'] += 1

                sent = last_sent + this_sent
                bias = len(last_sent)
                reference_obj_flag = False
                tmp_acts = []
                for k in range(len(acts)):
                    act_idx = acts[k]['act_idx']
                    obj_idxs = acts[k]['obj_idxs']
                    tmp_act_idx = act_idx - bias
                    if tmp_act_idx < 0:
                        log['act_reference_1'] += 1
                        #continue
                    tmp_obj_idxs = [[],[]]
                    for l in range(2):
                        for oi in obj_idxs[l]:
                            if oi == -1:
                                tmp_obj_idxs[l].append(oi)
                            else:
                                tmp_obj_idxs[l].append(oi - bias)
                                if oi - bias < 0:
                                    reference_obj_flag = True
                        assert len(tmp_obj_idxs[l]) == len(obj_idxs[l])
                    tmp_act_type = acts[k]['act_type']
                    tmp_related_acts = []
                    if len(acts[k]['related_acts']) > 0:
                        for idx in acts[k]['related_acts']:
                            tmp_related_acts.append(idx - bias)
                            if idx - bias < 0:
                                reference_related_acts = True
                                log['related_act_reference_1'] += 1
                        assert len(tmp_related_acts) == len(acts[k]['related_acts'])
                    tmp_acts.append({'act_idx': tmp_act_idx, 'obj_idxs': tmp_obj_idxs,
                                'act_type': tmp_act_type, 'related_acts': tmp_related_acts})
                assert len(tmp_acts) == len(acts)

                if j == 0:
                    if reference_obj_flag:
                        log['obj_reference_1'] += 1
                        for ii in range(len(words), len(words)+len(last_sent)):
                            word2sent[ii] = len(sents)
                        words.extend(last_sent)
                        sents.append(last_sent)
                        sent_acts.append({})
                    #elif reference_related_acts: 
                    #    log['related_act_reference_1'] += 1
                    else:
                        if len(last_sent) > 0:
                            log['non-obj_reference_1'] += 1
                            last_sent = []
                            bias = len(last_sent)
                            sent = last_sent + this_sent
                            acts = tmp_acts

                add_bias = len(words)
                for ii in range(len(words), len(words)+len(this_sent)):
                    word2sent[ii] = len(sents)
                words.extend(this_sent)
                sents.append(this_sent)
                sent_acts.append(acts)

                for k in range(len(tmp_acts)):
                    act_idx = tmp_acts[k]['act_idx']
                    obj_idxs = tmp_acts[k]['obj_idxs']
                    text_act_idx = act_idx + add_bias
                    if sent[act_idx + bias] != words[act_idx + add_bias]:
                        ipdb.set_trace()
                        print(sent[act_idx + bias], words[act_idx + add_bias])
                    text_obj_idxs = [[],[]]
                    for l in range(2):
                        for oi in obj_idxs[l]:
                            if oi == -1:
                                text_obj_idxs[l].append(-1)
                            else:
                                text_obj_idxs[l].append(oi + add_bias)
                                if sent[oi + bias] != words[oi + add_bias]:
                                    ipdb.set_trace()
                                    print(sent[oi + bias], words[oi + add_bias])
                        assert len(text_obj_idxs[l]) == len(obj_idxs[l])
                    text_act_type = tmp_acts[k]['act_type']
                    text_related_acts = []
                    if len(tmp_acts[k]['related_acts']) > 0:
                        for idx in tmp_acts[k]['related_acts']:
                            text_related_acts.append(idx + add_bias)
                        assert len(text_related_acts) == len(tmp_acts[k]['related_acts'])
                    text_acts.append({'act_idx': text_act_idx, 'obj_idxs': text_obj_idxs,
                                'act_type': text_act_type, 'related_acts': text_related_acts})

            assert len(word2sent) == len(words)
            assert len(sents) == len(sent_acts)
            if reference_related_acts:
                for m, a in enumerate(text_acts):
                    print('{}\t{}\n'.format(m, a))
                #ipdb.set_trace()
            data.append({'words': words, 'acts': text_acts, 'sent_acts': sent_acts,
                        'sents': sents, 'word2sent': word2sent})

        upper_bound = 0
        lower_bound = 0
        for d in data:
            for n in range(len(d['acts'])):
                act = d['acts'][n]['act_idx']
                objs = d['acts'][n]['obj_idxs']
                for l in range(2):
                    for obj in objs[l]:
                        if obj == -1:
                            continue
                        if obj - act < lower_bound:
                            lower_bound = obj - act
                            print(act, obj)
                        if obj - act > upper_bound:
                            upper_bound = obj - act
                            print(act, obj)
        print('\nupper_bound: {}\tlower_bound: {}\nlog history: {}\n'.format(
            upper_bound, lower_bound, log))

        with open('%s/%s_labeled_text_data.pkl'%(name, name), 'wb') as f:
            pickle.dump(data, f, protocol=2)


    def split_sents(self):
        num = 1
        ipdb.set_trace()
        texts = []
        #for fname in os.listdir(self.source):
        for i in range(self.num_texts):
            fname = '%d.txt' % (i + 1)
            if not fname.endswith('.txt'):
                continue
            with open(self.source + fname) as f:
                if self.one_line_data:
                    atext = f.read() #f.readlines()
                    #atext = re.sub(r'\.\n|\?\n|\!\n', '\n', atext)
                    #for j in range(len(btext)):
                    btext = atext.split('\n')#[:-1]
                    assert len(btext[-1]) != 0
                    #btext[0] = btext[0][0] + btext[0][1:].lower()
                    texts.append(btext)
                else:
                    text = f.read()
                    #atext = re.sub(r'\n|\r|,|;|\(|\)', ' ', text)
                    atext = re.sub(r'\n|\r|\(|\)', ' ', text)
                    btext = re.split(r'\. |\? |\! ', atext)
                    texts.append(btext[:-1])
                with open(self.out_path + '%d.txt' % num, 'w') as f1:
                    print(num)
                    f1.write('\n'.join(btext))
                    num += 1

        with open(self.save_file, 'wb') as outfile:
            pickle.dump(texts, outfile, protocol=2)


    def text_labeling(self):
        if self.home == 'wikihow':
            self.num_texts = 256
            self.save_file = 'wikihow/home_and_garden_500_words.pkl'
        if self.home != 'cooking':
            with open(self.save_file, 'rb') as f:
                texts = pickle.load(f)
                if self.home == 'wikihow':
                    _ = texts.pop(87) # skip out of place texts
                    _ = texts.pop(108)
                    _ = texts.pop(118)
                    _ = texts.pop(118)
                    _ = texts.pop(122)
                    _ = texts.pop(123)
                    _ = texts.pop(126)
                    _ = texts.pop(126)

        if os.path.exists(self.save_labeled_data):
            with open(self.save_labeled_data, 'rb') as f:
                print('Load data from %s...\n' % self.save_labeled_data)
                last_text, last_sent, data = pickle.load(f)
                print('last_text: %d\t last_sent: %d\n' % (last_text, last_sent))
            while True:
                init = input('Input last text num and sent num\n')
                if not init:
                    print('No input, program exit!\n')
                if len(init.split()) == 2:
                    start_text = int(init.split()[0])
                    start_sent = int(init.split()[1])
                    break
            # for i in range(len(data)):
            #     for j in range(len(data[i])):
            #         if len(data[i][j]) == 0:
            #             print(i, j)
            ipdb.set_trace()
        else:
            start_text = start_sent = 0
            data = [[] for _ in range(self.num_texts)]
        
        for i in range(start_text, self.num_texts):
            if self.home == 'cooking' and i >= 96:
                text = open('cooking/new_texts/%d.txt'%(i+1)).read()
                text = re.sub(r',|;', ' ', text)
                text = [t for t in text.split('\n') if len(t.split()) > 1]
            else:
                text = [t for t in texts[i] if len(t.split()) > 1]
            sents_num = len(text)
            print('\ntext %d: total %d words\n' % (i, sum([len(t.split()) for t in text])))
            if len(data[i]) > 0: #self.home != 'cooking' and i == start_text and 
                sents = data[i]
            else:
                sents = [{} for _ in range(sents_num)]
            try:
                if i != start_text:
                    start_sent = 0       
                for j in range(start_sent, sents_num):
                    #if len(data[i]) <= j or len(data[i][j]) > 0:
                    #    continue
                    sent = {}
                    this_sent = text[j].split() 
                    if j > 0: # print two sentences, used for coreference resolution
                        last_sent = text[j - 1].split()
                    else:
                        last_sent = []
                    if self.home == 'win2k':
                        this_sent[0] = this_sent[0].title()
                        if len(last_sent) > 0:
                            last_sent[0] = last_sent[0].title()
                    sent['last_sent'] = last_sent
                    sent['this_sent'] = this_sent
                    sent['acts'] = []
                    words = last_sent + this_sent
                    words_num = len(words)
                    print('T%d of %d, S%d of %d:' % (i, self.num_texts, j, sents_num))
                    for l, w in enumerate(words):
                        print('%s(%d)'%(w, l), end=' ')
                    while True:
                        act = input('\nInput an action and object indices:\n')
                        if not act:
                            break
                        if self.home == 'win2k2':
                            lest_input = 1
                            act_i = 0
                            obj_i = 1
                        else:
                            lest_input = 2
                            act_i = 1
                            obj_i = 2
                        if len(act.split()) <= lest_input:
                            if act == 'q':
                                raise QuitProgram()
                            elif act == 'r': # revise a sent
                                print(' '.join(sent['this_sent']))
                                text[j] = input('Input right sentence\n')
                                sent['this_sent'] = text[j].strip().split()
                                words = last_sent + sent['this_sent']
                                words_num = len(words)
                                for l, w in enumerate(words):
                                    print('%s(%d)'%(w, l), end=' ')
                                continue
                            else:
                                continue
                        nums = [int(a) for a in act.split()]
                        if self.home == 'win2k2':
                            act_type = 1
                            related_acts = []
                        else:
                            act_type = nums[0]
                            if act_type not in [1, 2, 3]: # essential, optional, exclusive
                                print('Wrong act_type!')
                                continue
                            if act_type == 3:
                                related_acts = input('Enter its related actions (indices):\n')
                                related_acts = [int(r) for r in related_acts.split()]
                                if len(related_acts) == 0:
                                    print('You should input related_acts!\n')
                                    continue
                                print('\tRelated actions: {}'.format([words[idx] for idx in related_acts]))
                            else:
                                related_acts = []

                        act_idx = nums[act_i]
                        if act_idx >= words_num:
                            print('action index %d out of range' % act_idx)
                            continue
                        obj_idxs = []
                        continue_flag = False
                        for idx in nums[obj_i: ]:
                            if idx >= words_num:
                                print('object index %d out of range' % idx)
                                continue_flag = True
                                break
                            obj_idxs.append(idx)
                        if continue_flag:
                            continue
                        obj_names = []
                        for k in obj_idxs:
                            if k >= 0:
                                obj_names.append(words[k])
                            else:
                                obj_names.append('NULL')
                        print('\t%s(%s)  act_type: %d' % (words[act_idx], ','.join(obj_names), act_type))
                        sent['acts'].append({'act_idx': act_idx, 'obj_idxs': obj_idxs,
                                            'act_type': act_type, 'related_acts': related_acts})
                    if len(sents) < sents_num:
                        sents.append({})
                    sents[j] = sent
            except Exception as e:
                print('Error:',e)
                if len(data) < self.num_texts:
                    data.append([])
                data[i] = sents
                with open(self.save_labeled_data, 'wb') as f:
                    pickle.dump([i, j, data], f)
                    break_flag = True
                    print('last_text: %d\t last_sent: %d\n' % (i, j))
                    break
            if len(data) < self.num_texts:
                    data.append([])
            data[i] = sents
        with open(self.save_labeled_data, 'wb') as f:
            pickle.dump([i, j, data], f, protocol=2)
            break_flag = True
            print('last_text: %d\t last_sent: %d\n' % (i, j))
        


if __name__ == '__main__':
    start = time.time()
    model = DataLabeler()
    #model.find_top_or_text_by_category()
    #model.add_object_type()
    #model.transfer('wikihow')
    model.text_labeling()
    #for name in ['win2k', 'wikihow', 'cooking']:
    #    model.transfer(name)
    end = time.time()
    print('Total time cost: %.2fs\n' % (end - start))

