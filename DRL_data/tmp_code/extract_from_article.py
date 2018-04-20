#coding:utf-8
import re
import time

f = open('split_windows2k_dataset.txt','w+')
for line in open('windows2k_dataset.txt').readlines():
    if re.search(r'!|@',line) or len(line) <= 1:
        pass
    else:
        #f.write(str(count)+' length: '+str(len(line))+'  '+re.sub(r'<b>|</b>','',line))
        f.write(re.sub(r'<b>|</b>|#|/|\\|,|\(|\)','',line))
f.close()
'''
count = 0
f = open('out_articles.txt','w+')
for line in open('articles.txt').readlines():
    count += 1 
    if re.search(r'# |##|#/|@',line) or len(line) <= 1:
        pass
    else:
        #f.write(str(count)+' length: '+str(len(line))+'  '+re.sub(r'<b>|</b>','',line))
        f.write(re.sub(r'<b>|</b>|#','',line))
print 'Done!'


sents = []
f = open('split_articles.txt','w+')
for line in open('out_articles.txt').readlines():
    temp_sent = re.split(r', |\n| ',line)
    sents.append(temp_sent)
    f.write(' '.join(temp_sent) + '\n') 
f.close()
'''
index = 19
words = 0
sents = 0
f = open(str(index)+'.txt','w+')
for line in open('split_windows2k_dataset.txt').readlines():
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
