#coding:utf-8
import mysql.connector
import time
import re

def div_sents(home_dir,i):
    wt = []
    for line in open(home_dir+str(i+1)+".txt"):
        st = line.split() #re.split(r' ',re.sub(r'\n','',line))
        #print st
        wt.append(st)
    return wt
    
home_dir = "/home/fengwf/Documents/DRL_data/long_texts/"
total_text = 128
start_time = time.clock()
stop = False
db = mysql.connector.connect(user='fengwf',password='123',database='test')
cur = db.cursor()
'''
create table tag_actions5(text_num int, sent_num int, sent varchar(400), tag_sent varchar(400));
'''
start_text,start_sent = re.split(r'[, ]',raw_input("Input text_num and sent_num:\n"))
start_text = int(start_text)
while (start_text < 0 or start_text > total_text-1):
    start_text,start_sent = re.split(r'[, ]',raw_input("Text_num out of range. Please input again:\n"))
    start_text = int(start_text)
#print 'start_text,type(start_sent)',start_text,type(start_sent)
for i in range(start_text,total_text):
    wt = div_sents(home_dir,i)
    #print wt
    #print sum(len(s) for s in wt)
    #assert 1 == 0
    print "\n\n***** Text name: %s *****" % (home_dir+str(i+1)+'.txt')
    print "Total sentences: %d          Total words: %d" % (len(wt),sum(len(s) for s in wt))
    #assert 1 == 0
    if i == start_text:
        start_sent = int(start_sent)
        if start_sent < 0 or start_sent >= len(wt):
            start_sent = 0
    else:
        start_sent = 0
    for ii in range(start_sent,len(wt)):
        
        display_sent = []
        max_tags = len(wt[ii])
        tag_sent = ['0' for col in range(max_tags)]
        for jj in range(max_tags):
            display_sent.append(wt[ii][jj]+'('+str(jj+1)+')')
        print '\nSentence',ii,':',' '.join(display_sent)  #输出每个单词及其序号
        temp_in = re.split(r'[, ]',raw_input("\nInput the total actions and their indices:\n-1 for exit\n-2 for search or delete\n"))
        if not temp_in[0]:
            num = 0
        else:
            num = int(temp_in[0])
            if num == -1:
                sent_num = ii
                stop = True
                break
            tags = [int(tag) for tag in temp_in[1:]]
            if num != -2 and num != len(tags):
                num = -3
                tags = []
        while(num <-2 or num > max_tags):
            temp_in = re.split(r'[, ]',raw_input("Wrong num, please input again:\n"))
            print 'temp_in',temp_in
            if not temp_in[0]:
                num = 0
            else:
                num = int(temp_in[0])
                if num == -1:
                    sent_num = ii
                    stop = True
                    break
                elif num == -2:
                    break
                else:
                    tags = [int(tag) for tag in temp_in[1:]]
                    if num != len(tags):
                        num = -3
                        tags = []
        if num == -2:#查询或删除数据
            cur.execute('select * from tag_actions2')#选择table tag_actions2
            result = cur.fetchall()
            print "\nTotal records: ",len(result)
            n = raw_input("How many records do you want to display: \n A for all records \n L for last ten records \n D for delete last record\n ")
            n = n.upper()
            while True:
                if not n:#什么都没输入，直接跳
                    break
                if n not in ['A','L','D']:
                    n = raw_input("Please input A, L or D: ")
                    n = n.upper()
                else:
                    break
            if n == 'A':
                print "\n-------------------------All Records In The Database-------------------------"
                print "Total records: ",len(result)
                for detail in result:
                    print '\n',detail
                print "\nTotal records: ",len(result)
                print "-------------------------------End Of Display--------------------------------"
            elif n == 'L':
                print "\n----------------------------Last Ten Records-------------------------------"
                if len(result) >= 10:
                    start = len(result)-10
                else:
                    start = 0
                for di in range(start,len(result)):
                    print '\n',result[di]
                print "\n-----------------------------End Of Display--------------------------------"
            elif n == 'D':
                tn,sn = re.split(r'[, ]',raw_input("Input text_num and sent_num to delete:\n"))
                delete = "delete from tag_actions2 where text_num="+tn+" and sent_num="+sn
                print '\ndelete',delete,'\n'
                cur.execute(delete)
                db.commit()
                cur.execute('select * from tag_actions2')#选择table tag_actions2
                result = cur.fetchall()
                print "\n----------------------------Last Ten Records-------------------------------"
                if len(result) >= 10:
                    start = len(result)-10
                else:
                    start = 0
                for di in range(start,len(result)):
                    print '\n',result[di]
                print "\n-----------------------------End Of Display--------------------------------"
            print '\nSentence',ii,':',' '.join(display_sent)  #输出每个单词及其序号
            temp_in = re.split(r'[, ]',raw_input("\nInput the total actions and their indices:\n-1 for exit\n-2 for search or delete\n"))
            if not temp_in[0]:
                num = 0
            else:
                num = int(temp_in[0])
                if num == -1:
                    sent_num = ii
                    stop = True
                    break
                tags = [int(tag) for tag in temp_in[1:]]
                if num != len(tags):
                    num = -3
                    tags = []
            while(num <-1 or num >max_tags):
                temp_in = re.split(r'[, ]',raw_input("Wrong num too, please input again:\n"))
                if not temp_in[0]:
                    num = 0
                else:
                    num = int(temp_in[0])
                    if num == -1:
                        sent_num = ii
                        stop = True
                        break
                    tags = [int(tag) for tag in temp_in[1:]]
                    if num != len(tags):
                        num = -3
                        tags = []
        if stop:
            break
        if num == 0:
            print "\n----------No actions to be tag----------\n"
            '''
            insert = "insert into tag_actions2 values("+str(i)+','+str(ii)+',"'+' '.join(wt[ii])\
                +'","'+' '.join(tag_sent)+'")'
            print 'Tagged sentences: [{}]'.format(' '.join(tag_sent))
            cur.execute(insert)
            db.commit()
            '''
        if num == -1:#终止操作
            sent_num = ii
            stop = True
            break
        if num > 0:#标记动作
            for kk in range(num):
                tag_sent[tags[kk]-1] = '1'  #标记动作
                print 'Tagged actions:',wt[ii][tags[kk]-1],tag_sent[tags[kk]-1]
            insert = "insert into tag_actions2 values("+str(i)+','+str(ii)+',"'+' '.join(wt[ii])\
                +'","'+' '.join(tag_sent)+'")'
            print 'Tagged sentences: [{}]'.format(' '.join(tag_sent))
            cur.execute(insert)
            db.commit()
    if stop:
        print '\nText_num: %d; Sentence_num: %d' %(i,sent_num)
        break
end_time = time.clock()
print  'Total time cost: %ds'%(end_time-start_time)