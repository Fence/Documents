#coding:utf-8
import re
'''
f = open('out_paragraph.txt','w+')
for line in open('paragraph.instructions').readlines():
    if re.search(r'Cleaned.+|{}',line) or len(line) <= 1:
        pass
    else:
        #print line
        sents = re.sub(r',','',line)
        sents = re.sub(r'  |\n',' ',sents)
        sents = re.split(r'\. ',sents)
        #print sents
        for s in sents:
            if s:
                f.write(s+'\n')
f.close()
'''
index = 1
words = 0
sents = 0
f = open(str(index)+'.txt','w+')
for line in open('out_paragraph.txt').readlines():
    words += len(line.split())
    if words >= 500:
        print '-----Total words in text %d is %d-----' %(index,words-len(line.split()))
        f.close()
        index += 1
        words = len(line.split())
        sents = 0
        f =open(str(index)+'.txt','w+')
    f.write(line)
    sents += 1
    print 'Writing in text %d sentences %d' %(index,sents)
