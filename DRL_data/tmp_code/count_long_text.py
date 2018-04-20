import re
import mysql.connector

db = mysql.connector.connect(user='fengwf', password='123', database='test')
cur = db.cursor()
result = []
count = []
table_name = 'tag_actions5'
max_text_num = 35
for i in range(max_text_num):
    get_data = "select * from " + table_name + " where text_num = " + \
    str(i) + " order by sent_num"
    #print 'get_data = %s'%get_data
    print 'text num : %d'%i
    cur.execute(get_data)
    temp = cur.fetchall()
    result.append(temp)
    num = 1
    words = 0
    for j in range(len(temp)):
        sent = temp[j][2].split()
        ws = len(sent)
        if ws == 1 and sent[0] == '-----LONG_TEXT_SEPARATOR-----':
            num += 1
            #print temp[j][2].split()[0]
        else:
            words += ws
    print 'num = %d'%num
    print 'total words: %d\n'%words
    if words > 500:
        print '----- words > 500 -----'
    count.append(num)
print count
print 'count = %d'%sum(count)