#coding:utf-8
import re
import time
import nltk
import mysql.connector
import numpy as np

from nltk.parse.stanford import StanfordDependencyParser, StanfordParser
from nltk.tag import StanfordPOSTagger

class EANLP:
    def __init__(self):
        print '\nStarting EANLP...'
        self.start = time.time()

    def data_processing(self, actionDB, num):
        result =  []
        db = mysql.connector.connect(user='fengwf',password='123',database='test')
        cur = db.cursor()
        get_data = "select * from " + actionDB + " where text_num < " + str(num)
        if actionDB == 'tag_actions':
            get_data += ' and text_num != 30 ' 
        cur.execute(get_data)
        result.extend(cur.fetchall())
        #print 'len(result)',len(result)

        tags = []
        sents = []
        for i in range(len(result)):
            sent_lower = result[i][2][0].lower() + result[i][2][1:]  #首字母换回小写
            words_of_sent = sent_lower #re.split(r' ',sent_lower)
            temp_tags_of_sent = re.split(r' ',result[i][3])
            tags_of_sent = [int(t) for t in temp_tags_of_sent]  #是动作则标注为1，不是则为0
            
            sents.append(words_of_sent) #每个句子的单词作为一个list
            tags.append(tags_of_sent)
        
        print 'total sentences: %d'%len(sents)
        print 'total words: %d'%sum([len(w.split()) for w in sents])
        return sents, tags


    def RegexpParser_init(self):
        grammar=r"""VP:{<NN|NNS|PRP|NNP|NNPS><RB>?<MD><RB>?<VB><RB>?<VBD|VBG|VBN>}
        VP:{<NN|NNS|PRP|NNP|NNPS><RB>?<VBP><RB>?<VBN><RB>?<VBD|VBN>}
        VP:{<NN|NNS|PRP|NNP|NNPS><RB>?<VBZ><RB>?<VBD|VBG|VBN>}
        VP:{<VB|VBD|VBG|VBN|VBP><RB>?<PRP\$|DT>?<VBN|VBD|JJ>?<NN|NNS|NNP|NNPS>+}
        VP:{<VB|VBD|VBG|VBN|VBP><RB>?<PRP\$|DT>?<VBN|VBD|JJ>?<PRP>}
        VP:{<VB|VBD|VBG|VBN|VBP><IN><DT>?<NN|NNS>+}
        VP:{<VB|VBD|VBG|VBN|VBP><IN>?<DT>?<CD>?<VBN|VBD|JJ>?<NN|NNS|NNP|NNPS>+}
        VP:{<VB|VBD|VBG|VBN|VBP><IN>?<DT>?<CD>?<VBN|VBD|JJ>?<PRP>}
        VP:{<VB|VBD|VBG|VBN|VBP><IN><VBN|VBD><NN|NNS>+}
        """#被动句中去掉了WDT(Wh-determiner)
        #主动句中只考虑了动词原形的//VP:{<VB><DT>?<VBN|VBD|JJ>?<NN|NNS|NNP|NNPS>+}只有一个代词的时候不重复
        #something can be done
        #something has been done
        #something is/was done
        #do/did something, Set your oven, put the mixture in the oven.
        #Ensure the twists have extended above the plate by
        #Get a ten inch pie plate, add in the hash browns
        #add in something
        #
        self.cp = nltk.RegexpParser(grammar)  #生成语法块
        self.pron_tag = ["it", "they", "them", "this", "that", "these", "those"]
        self.noun_tag = ['NN','NNS','NNP','NNPS','PRP']
        self.verb_tag = ['VB','VBD','VBG','VBN','VBP']


    def StanfordParser_init(self):
        pos_tagger_jar = '/home/fengwf/stanford/postagger/stanford-postagger.jar'
        pos_tagger_models = '/home/fengwf/stanford/postagger/models/english-bidirectional-distsim.tagger'
        path_to_jar = '/home/fengwf/stanford/stanford-corenlp-3.7.0.jar'
        path_to_models_jar = '/home/fengwf/stanford/english-models.jar'
        self.dep_parser = StanfordDependencyParser(
            path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
        self.std_parser = StanfordParser(
            path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
        self.pos_tagger = StanfordPOSTagger(
            pos_tagger_models, pos_tagger_jar)


    def regexp_find_vp(self, sent):
        temp_myvp = []
        obj = []
        result = self.cp.parse(sent)  
        b = re.findall(r'\(VP.+\)',str(result)) 
        for c in b:
            #d = re.findall(r'[\w|\-]+\$?',c)#所有的字母以及符号$
            d = re.findall(r'[A-Za-z-\']+\$?|\d+/\d+',c)#所有的字母以及符号$，don't，20/20这些类型的
            if len(d) % 2 == 0:#去除分错的
                continue
            #if i ==1454:  print d
            #print d[1],d[len(d)-2]
            if d[2] in self.verb_tag and d[3] in ['if']:
                continue
            if d[2] in self.verb_tag:#主动句，第一个单词是动词
                if d[4] == 'NN' and d[3] == 't':#排除don't等缩写的影响
                    pass
                else:
                    j = 4
                    obj.append(d[1])
                    while j < len(d):
                        if d[j] in self.noun_tag:
                            #print 'd[j]: ',d[j]
                            obj.append(d[j-1])
                        j += 2
                    #print 'dobj:',obj
                    temp_myvp.append(obj)
                    obj = []
            elif d[2] in self.pron_tag:#被动句，第一个单词是名词性主语
                obj = [d[len(d)-2],d[1]]
                #print 'idobj:',obj
                temp_myvp.append(obj)
                obj = []
        return temp_myvp


    def stanford_find_vp(self, sent, f):
        stf_vp = []
        if len(sent.split()) <= 1:
            return stf_vp
        result = self.dep_parser.raw_parse(sent)
        dep = result.next()
        for t in list(dep.triples()):
            assert  len(t) == 3
            relation = str(t[1])
            if relation in ['dobj', 'nsubjpass']:
                word1 = str(t[0][0])
                tag1 = str(t[0][1])
                word2 = str(t[2][0])
                tag2 = str(t[2][1])
                f.write('\n%s:  %s ( %s )'%(relation, word1, word2))
                stf_vp.append([word1, tag1])
        return stf_vp



    def main(self, actionDB, text_num, f):
        sents, label = self.data_processing(actionDB, text_num)
        f.write('Tagging for total %d sentences with nltk methods'%len(sents))

        tags = []
        tag_act = []
        temp_tag = []
        temp_tag_act = []
        self.StanfordParser_init()
        self.RegexpParser_init()
        for i in range(len(sents)):
            print '\nTable: %s,    Sentence: %d'%(actionDB, i)
            sent = re.split(r' ', sents[i])
            f.write('\n\nSentence: %d'%(i+1))
            tagged = nltk.pos_tag(sent)
            #print '\n',tagged
            f.write('\ntagged_sent: '+str(tagged))
            myvp = self.regexp_find_vp(tagged)
            #print 'myvp:',myvp
            f.write('\nmyvp: '+str(myvp))
            stf_vp = self.stanford_find_vp(sents[i], f)
            #print 'stf_vp',stf_vp

            maxlen = len(myvp)
            count = 0
            stf_len = len(stf_vp)
            ind = 0
            for (w,t) in tagged:
                if stf_len > 0 and ind < stf_len:
                    if stf_vp[ind][0] == w:
                        ind += 1
                        temp_tag.append(1)
                        temp_tag_act.append(w)
                        continue
                        #print '-----tagging form stf_vp-----'
                if maxlen > 0 and count < maxlen:
                    if len(myvp[count]) > 0:
                        if myvp[count][0] == w:
                            count += 1
                            temp_tag.append(1)
                            temp_tag_act.append(w)
                            continue
                            #print '-----tagging form myvp-----'
                temp_tag.append(0)
                temp_tag_act.append('')
            #print temp_tag
            #print temp_tag_act
            f.write('\nlabel:    '+str(label[i]))
            f.write('\ntemp_tag: '+str(temp_tag))
            f.write('\ntemp_tag_act: '+str(temp_tag_act))
            tags.append(temp_tag)
            tag_act.append(temp_tag_act)
            temp_tag = []
            temp_tag_act = []


        total_actions = sum([sum(l) for l in label])
        right_actions = 0
        tag_actions = 0
        precision = 0
        recall = 0
        f_measure = 0
        if len(label) != len(tags):
            print '-----len(label) = %d,  len(tags) = %d-----'%(len(label),len(tags))
            return
        for j in range(len(label)):
            if len(label[j]) != len(tags[j]):
                print '+++++len(label[%d]) = %d,  len(tags[%d]) = %d+++++'%\
                (j, len(label), j, len(tags))
                continue
            for k in range(len(label[j])):
                if tags[j][k] == 1:
                    tag_actions += 1
                    if label[j][k] == 1:
                        right_actions += 1
        if total_actions > 0 and tag_actions > 0:
            recall = float(right_actions)/total_actions
            precision = float(right_actions)/tag_actions
        else:
            recall = 0.0
            precision = 0.0
        if (recall+precision) > 0:
            f_measure = 2*recall*precision/(recall+precision)
        else:
            f_measure = 0.0
        total_words = sum([len(re.split(r' ', w)) for w in sents])
        #print total_words
        accuracy = float(total_words - total_actions - tag_actions + 2*right_actions)/total_words
        f.write('\n\n\n\ntotal_actions: %d'%total_actions)
        f.write('\nright_actions: %d'%right_actions)
        f.write('\ntag_actions: %d'%tag_actions)
        f.write('\nrecall: %f'%recall)
        f.write('\nprecision: %f'%precision)
        f.write('\nf_measure: %f'%f_measure )
        f.write('\naccuracy: %f'%accuracy )
        print '\ntotal_actions: %d'%total_actions
        print 'right_actions: %d'%right_actions
        print 'tag_actions: %d'%tag_actions
        print 'recall: %f'%recall
        print 'precision: %f'%precision
        print 'f_measure: %f'%f_measure 
        print '\naccuracy: %f'%accuracy 
                    

        self.end = time.time()
        print "Time cost %ds" %(self.end-self.start)
        f.write("\nTime cost %ds" %(self.end-self.start))


if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('gb18030')
    actionDB = ["tag_actions", "tag_actions1", "tag_actions2", "tag_actions3",
    "tag_actions4", "tag_actions5", "tag_actions6"]
    num = [64, 52, 33, 54, 111, 35, 43]
    test_num = [8, 8, 8, 8, 8, 8, 8]
    test = EANLP()
    for i,table in enumerate(actionDB):
        f = open('./nltk_results/'+table+'_tn8.txt','w+')
        test.main(table, test_num[i], f)
        f.close()