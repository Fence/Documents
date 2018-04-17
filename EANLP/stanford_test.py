#coding:utf-8
import re
import os
import time
import nltk
import pprint
import mysql.connector
import numpy as np
from find_vp_myself import find_vp_single_sent
from nltk.parse.stanford import StanfordDependencyParser, StanfordParser
from nltk.tag import StanfordPOSTagger

pos_tagger_jar = '/home/fengwf/stanford/postagger/stanford-postagger.jar'
pos_tagger_models = '/home/fengwf/stanford/postagger/models/english-bidirectional-distsim.tagger'
path_to_jar = '/home/fengwf/stanford/stanford-corenlp-3.7.0.jar'
path_to_models_jar = '/home/fengwf/stanford/english-models.jar'
dep_parser = StanfordDependencyParser(
    path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
std_parser = StanfordParser(
    path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
pos_tagger = StanfordPOSTagger(
    pos_tagger_models, pos_tagger_jar)


def processing_data(actionDB, num):
    print "Processing data.py function processing_data..."

    result =  []
    db = mysql.connector.connect(user='fengwf', password='123', database='test')
    cur = db.cursor()
    get_data = "select * from " + actionDB + " where text_num = " + str(num)
    cur.execute(get_data)
    result.extend(cur.fetchall())
    print 'len(result)',len(result)

    tags = []
    words = []
    for i in range(len(result)):
        #sent_lower = result[i][2][0].lower() + result[i][2][1:]  #首字母换回小写
        #words_of_sent = sent_lower #re.split(r' ',sent_lower)
        words_of_sent = result[i][2]
        temp_tags_of_sent = re.split(r' ',result[i][3])
        tags_of_sent = [int(t) for t in temp_tags_of_sent]  #是动作则标注为1，不是则为0
        
        words.append(words_of_sent) #每个句子的单词作为一个list
        tags.append(tags_of_sent)
    
    print 'total sentences: %d'%len(words)
    print 'total words: %d'%sum([len(w.split()) for w in words])
    return words, tags


def dep_parse_dbs(actionDBs, max_text_num):
    for ind in range(len(actionDBs)):
        print '\n\nprocessing actionDB %s ...'%actionDBs[ind]
        f = open('./dependency_results/'+actionDBs[ind]+'.txt','w+')
        f.write('actionDB: %s'%actionDBs[ind])
        
        for num in range(max_text_num[ind]):
            print '\nprocessing text %d of actionDB %s ...'%(num,actionDBs[ind])
            if num == 30 and ind == 0:
                continue
            raw_sents, labels = processing_data(actionDBs[ind], num)
            f.write('\n\ntext_num: %d '%num)
            f.write('total sentences: %d'%len(raw_sents))
            f.write('  total words: %d'%sum([len(w.split()) for w in raw_sents]))
            sents = dep_parser.raw_parse_sents(raw_sents)

            for i in range(len(raw_sents)):
                result = sents.next()
                dep = result.next()
                #if i%20 == 0:
                #    print '\nSentence %d: %s'%(i,raw_sents[i])
                f.write('\n\nSentence %d: %s'%(i, raw_sents[i]))
                for t in list(dep.triples()):
                    assert  len(t) == 3
                    word1 = str(t[0][0])
                    tag1 = str(t[0][1])
                    relation = str(t[1])
                    word2 = str(t[2][0])
                    tag2 = str(t[2][1])
                    f.write('\n%s  %s  %s  %s  %s'%\
                        (word1, tag1, relation, word2, tag2))
        f.close()


def dep_parse_db(actionDB, max_text_num):
    print '\n\nprocessing actionDB %s ...'%actionDB
    f = open('./dependency_results/'+actionDB+'.out','w+')
    f.write('actionDB: %s'%actionDB)
    
    for num in range(max_text_num):
        if actionDB == 'tag_actions' and num == 30:
            continue
        print '\nprocessing text %d of actionDB %s ...'%(num,actionDB)
        raw_sents, labels = processing_data(actionDB, num)
        f.write('\n\ntext_num: %d '%num)
        f.write('total sentences: %d'%len(raw_sents))
        f.write('  total words: %d'%sum([len(w.split()) for w in raw_sents]))
        sents = dep_parser.raw_parse_sents(raw_sents)

        for i in range(len(raw_sents)):
            result = sents.next()
            dep = result.next()
            f.write('\n\nSentence %d: %s'%(i, raw_sents[i]))
            for t in list(dep.triples()):
                assert  len(t) == 3
                relation = str(t[1])
                if relation in ['dobj', 'nsubjpass']:
                    word1 = str(t[0][0])
                    tag1 = str(t[0][1])
                    word2 = str(t[2][0])
                    tag2 = str(t[2][1])
                    f.write('\n%s:  %s ( %s )'%(relation, word1, word2))
    f.close()


def split_text():
    source = './sample_texts/'
    tdir = os.listdir(source)
    count = 0
    for d in tdir:
        if d.find('.txt '):
            print source + d
            count += 1
            f = open(source + str(count) + '.out', 'w+')
            raw_text = open(source+d).read()
            a = re.sub(r'\. ', '. \n', raw_text)
            b = re.sub(r'\? ', '? \n', a)
            c = re.sub(r'\! ', '! \n', b)
            f.write(c)
            f.close()


def create_sents_list(text):
    if not os.path.isfile(text):
        print '\nError: File "%s" does not exist!\n'%text
        return
    sents = []
    for line in open(text):
        a = line.split()
        if len(a) <= 1:
            continue
        sents.append(' '.join(a))

    return sents


def dep_parse_text(text, out_name):
    if not os.path.isfile(text):
        print '\nError: File "%s" does not exist!\n'%text
        return
    sents = open(text).read().split('\n')
    f = open(out_name, 'w+')
    print len(sents)
    count = 0
    for sent in sents:
        #print sent
        if len(sent.split()) <= 1:
            continue
        result = dep_parser.raw_parse(sent)
        dep = result.next()
        count += 1
        print count
        f.write('\n\nSentence %d: %s'%(count, sent))
        for t in list(dep.triples()):
            assert  len(t) == 3
            word1 = str(t[0][0])
            tag1 = str(t[0][1])
            relation = str(t[1])
            word2 = str(t[2][0])
            tag2 = str(t[2][1])
            #print '\n%s  %s  %s  %s  %s'%(word1, tag1, relation, word2, tag2)
            f.write('\n%s  %s  %s  %s  %s'%(word1, tag1, relation, word2, tag2))
            #f.write('\n%s:  %s ( %s )'%(relation, word1, word2))
    f.close()
        

def test():
    sent = """Get the right size tablecloth. \
    Using a tablecloth that is the right size is important when boxing a table. \
    A tablecloth that can't fully cover the table won't look quite right when finished. \
    Make sure you measure your table and use a tablecloth that fits to get the best results."""
    result = std_parser.raw_parse(sent)
    result1 = dep_parser.raw_parse(sent)
    result2 = pos_tagger.tag(sent.split())
    dep = result.next()
    dep1 = result1.next()
    print dep
    print list(dep1.triples())
    print result2


if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('gb18030')
    for i in range(5):
        dep_parse_text('./sample_texts/n'+str(i+1)+'.out', './test_'+str(i+1)+'.txt')


    '''
    for i in range(5):
        sents = create_sents_list('./sample_texts/' + str(i+1) + '.out')
        if len(sents):
            f = open('./sample_texts/n' + str(i+1) + '.out', 'w+')
            for s in sents:
                f.write(s+'\n')
            f.close()
    '''
    #split_text()
    #test()
    '''
    import sys
    reload(sys)
    sys.setdefaultencoding('gb18030')
    start = time.time()
    actionDBs = ["tag_actions","tag_actions1","tag_actions2",
    "tag_actions3","tag_actions4","tag_actions5"]
    max_text_num = [64,52,64,54,111,35]
    #dep_parse_dbs(actionDBs, max_text_num)
    for i in range(5):
        dep_parse_db(actionDBs[i], max_text_num[i])           
    end = time.time()
    print '\ntotal time cost: %fs'%(end-start)
    '''
