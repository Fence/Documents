#coding:utf-8
import wx
import os
import mysql.connector
import time
import re

'''
create table tag_actions6(text_num int, sent_num int, sent varchar(400), tag_sent varchar(400));
create table new_actions(text_num int, sent_num int, sent varchar(400), tag_sent varchar(400));
'''
#"/home/fengwf/Documents/DRL_data/long_texts/long_texts_ccfs/"
#'/home/fengwf/Documents/DRL_data/windows_help_support/win10_out/'
class TextFrame(wx.Frame):
    def __init__(self):
        self.home_dir = 'tag_actions1/'#"./DRL_data/windows_help_support/win10_out/"
        self.table = 'new_actions1'
        self.total_text = 52
        self.text_num = 0
        self.sent_num = 0
        self.num = 0
        self.max_tags = 0
        self.words = []
        self.tag_sent = []
        self.display_sent = []
        self.result = []
        self.start_flag = False
        self.search_flag = False
        self.delete_flag = False
        self.change_flag = False
        self.table_flag = False
        self.exit_flag = False
        self.show_frame()
        

    def show_frame(self):
        wx.Frame.__init__(self, None, -1, 'Tag Actions from texts', size=(800, 600))  #整个界面大小
        panel = wx.Panel(self, -1)
        panel.SetBackgroundColour('aquamarine')
        panel.Refresh()

        str = "Welcome to the tag_actions.app of Fence!"  #静态显示文本，欢迎标题
        self.text = wx.StaticText(panel, -1, str, (140,-1))
        font1 = wx.Font(18, wx.ROMAN, wx.ITALIC, wx.BOLD)  #字体18号，罗马字，斜体，加粗
        self.text.SetFont(font1)
        self.text.SetForegroundColour('tan')  #颜色为棕色
        
        self.info_text = wx.StaticText(
            panel, -1, "All rights reserved"+u'©'+"fengwf.PlanLab.2016", (540,540)) #输入文本框
        self.info_text.SetForegroundColour('dark slate blue')

        self.OutputText = wx.TextCtrl(panel, -1,"Input text_num and sent_num:\n", size=(700, 400), 
            pos=(50,40), style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2) #创建一个输出文本控件
        font2 = wx.Font(12, wx.NORMAL, wx.NORMAL, wx.NORMAL)
        self.OutputText.SetFont(font2)
        #self.OutputText.SetForegroundColour('orchid')
        self.pointer = self.OutputText.GetInsertionPoint()
        
        self.InputLabel = wx.StaticText(panel, -1, "Your input:", (100,455)) #输入文本框
        self.InputText = wx.TextCtrl(panel, -1, "", size=(175, -1), \
        style=wx.TE_PROCESS_ENTER, pos=(180,450))
        self.InputText.SetInsertionPoint(0)
        self.Bind(wx.EVT_TEXT_ENTER,self.Input_Box,self.InputText)
        
        self.SearchLabel = wx.StaticText(panel, -1, "Search input:", (390,455)) #输入文本框
        self.SearchText = wx.TextCtrl(panel, -1, "", size=(175, -1), \
        style=wx.TE_PROCESS_ENTER, pos=(480,450))  #查询的输入文本框
        self.SearchText.SetInsertionPoint(0)
        self.Bind(wx.EVT_TEXT_ENTER,self.Search_Input,self.SearchText)

        self.button1 = wx.Button(panel, -1,"Search", pos=(100, 500))  #搜索按钮
        self.Bind(wx.EVT_BUTTON, self.Search_Button, self.button1)
        self.button1.SetDefault()
        
        self.button2 = wx.Button(panel, -1,"A", pos=(180, 500))  #按钮，显示全部记录
        self.Bind(wx.EVT_BUTTON, self.ShowAll_Button, self.button2)
        self.button2.SetDefault()
        
        self.button3 = wx.Button(panel, -1,"L", pos=(260, 500))  #按钮，显示最近十条记录
        self.Bind(wx.EVT_BUTTON, self.ShowLast10_Button, self.button3)
        self.button3.SetDefault()
        
        self.button4 = wx.Button(panel, -1,"D", pos=(340, 500))  #按钮，删除记录
        self.Bind(wx.EVT_BUTTON, self.Delete_Button, self.button4)
        self.button4.SetDefault()
        
        self.button5 = wx.Button(panel, -1,"Exit", pos=(500, 500))  #按钮，退出
        self.Bind(wx.EVT_BUTTON, self.Exit_Button, self.button5)
        self.button5.SetDefault()
        
        self.button6 = wx.Button(panel, -1,"Reboot", pos=(420, 500))  #按钮，退出
        self.Bind(wx.EVT_BUTTON, self.Reboot_Button, self.button6)
        self.button6.SetDefault()
        
        self.button7 = wx.Button(panel, -1,"ChangeDir", pos=(580, 500))  #按钮，退出
        self.Bind(wx.EVT_BUTTON, self.Change_dir_Button, self.button7)
        self.button7.SetDefault()
        
    def div_sents(self):
        try:
            t = open(self.home_dir+str(self.text_num+1)+".txt").read() 
        except IOError:
            self.OutputText.AppendText('\Could not open, no such file or directory as: %s\n'\
                %self.home_dir)
        t = re.sub(r'\n',' ',t)
        t = re.sub(r',|"|\(|\)|\[|\]|\{|\}','',t)
        self.words = []
        sents = re.split(r'[\.?!:;]+ ',t)  #分成每个句子，分号冒号也算完成一个句子
        for i in sents:
            words = i.split()
            if len(words) > 0:
                words[0] = words[0].lower()
                self.words.append(words)
        return self.words


    def get_text_from_database(self):
        tmp_table = self.home_dir[:-1]
        get_data = "select * from " + tmp_table + " where text_num=" + \
            str(self.text_num) + " order by sent_num"
        cur.execute(get_data)
        result = cur.fetchall()
        assert len(result) > 0
        
        self.tags = []
        self.words = []
        for i in range(len(result)):
            #get sentences from database
            sent_lower = result[i][2][0].lower() + result[i][2][1:]
            words_of_sent = re.split(r' ',sent_lower)
            tags_of_sent = re.split(r' ',result[i][3])
            self.words.append(words_of_sent)
            self.tags.append(tags_of_sent)
        print '------------  len(words)',sum([len(i) for i in self.words])
    

    def set_output(self, message, color):
        self.pointer = self.OutputText.GetInsertionPoint()
        self.OutputText.AppendText(message)
        tp_point = self.pointer
        self.pointer = self.OutputText.GetInsertionPoint()
        self.OutputText.SetStyle(tp_point, self.pointer, wx.TextAttr(color, wx.NullColour))
    

    def Input_Box(self, event):
        input_str = self.InputText.GetValue().encode("ascii")
        temp_in = input_str.split()
        self.OutputText.AppendText(input_str+'\n')
        if not self.start_flag:
            if len(temp_in)!=2:
                message = '\nText_num out of range. Please input again:\n'
                self.set_output(message, "red")
            else:
                start_text = int(temp_in[0])
                start_sent = int(temp_in[1])
                if start_text < 0 or start_text > self.total_text-1:                    
                    message = '\nText_num out of range. Please input again:\n'
                    self.set_output(message, "red")
                else:
                    self.text_num = start_text
                    self.sent_num = start_sent
                    self.get_text_from_database()
                    self.start_flag = True
                    if self.sent_num >= len(self.words) or self.sent_num < 0:
                        self.sent_num = 0
                    message = "\n\n***** Text name: %s *****"%(self.home_dir+str(self.text_num)+'.txt')
                    message += "\nTotal sentences: %d\
                            Total words: %d\n" % (len(self.words),sum(len(s) for s in self.words))
                    self.set_output(message, "maroon")
                    self.display_sent = []
                    self.max_tags = len(self.words[self.sent_num])
                    self.tag_sent = self.tags[self.sent_num]
                    for jj in range(self.max_tags):
                        self.display_sent.append(self.words[self.sent_num][jj]+'('+str(jj)+')')
                    message = '\n\nSentence{}: {}\n'.format(self.sent_num, ' '.join(self.display_sent))
                    message += 'Current tags: %s\n'%' '.join(self.tags[self.sent_num])
                    self.set_output(message, "blue")
                    self.OutputText.AppendText("Input the total actions and their indices:\n")
        
        else:
            self.num = []
            tags = []
            if temp_in:
                if temp_in[0] == 'a': #修改整个句子的tag
                    if temp_in[1] == 'b': #修改整个句子的tag为全0
                        self.tag_sent = [0]*len(self.tag_sent)
                    else:
                        self.tag_sent = temp_in[1:]
                    self.OutputText.AppendText('Current tag: %s\n'%' '.join(self.tag_sent))
                    self.InputText.Clear()
                    return
                new_temp_in = input_str.split(',')
                assert len(new_temp_in) == 2
                tag0 = new_temp_in[0].split()  #否定标签
                tag1 = new_temp_in[1].split()  #动作短语标签
                self.num = [len(tag0), len(tag1)]
                tags.append([int(t) for t in tag0])
                tags.append([int(t) for t in tag1])

            if not self.num:
                self.OutputText.AppendText("\n----------No actions to be tag----------\n")
                insert = "insert into "+self.table+" values("+str(self.text_num)+','+\
                    str(self.sent_num)+',"'+' '.join(self.words[self.sent_num])+'","'+\
                    ' '.join(self.tag_sent)+'")'
                cur.execute(insert)
                db.commit()
            else:
                for kk in range(self.num[0]):
                    idx = tags[0][kk]
                    self.tag_sent[idx] = '2'  #标记否定动作
                    self.OutputText.AppendText('\nTagged negative action: {} {} {} {}'.format(
                        str(self.words[self.sent_num][idx]), str(self.words[self.sent_num][idx+1]), 
                        str(self.tag_sent[idx]), str(self.tag_sent[idx+1])))
                for kk in range(self.num[1]):
                    idx = tags[1][kk]
                    self.tag_sent[idx] = '3'  #标记动作短语
                    self.OutputText.AppendText('\nTagged action pair: {} {} {} {}'.format(
                        str(self.words[self.sent_num][idx-1]), str(self.words[self.sent_num][idx]), 
                        str(self.tag_sent[idx-1]), str(self.tag_sent[idx])))

                insert = 'insert into {} values({},{},"{}","{}")'.format(self.table, self.text_num, 
                    self.sent_num, ' '.join(self.words[self.sent_num]), ' '.join(self.tag_sent))
                self.OutputText.AppendText('\nTagged sentences: [{}]'.format(' '.join(self.tag_sent)))
                cur.execute(insert)
                db.commit()

            self.sent_num += 1
            if self.sent_num >= len(self.words):
                self.text_num += 1
                if self.text_num >= self.total_text:
                    end_time = time.time()
                    self.OutputText.AppendText('Total time cost: %ds'%(end_time-start_time))
                    print 'Total time cost: %ds'%(end_time-start_time)
                    wx.Exit()
                    self.exit_flag = True
                else:
                    self.sent_num = 0
                    self.get_text_from_database()
                    message = "\n\n\n------------------- A new text is starting ------------------\n"
                    message += "\n***** Text name: %s *****"%(self.home_dir+str(self.text_num)+'.txt')
                    message += "\nTotal sentences: %d\
                        Total words: %d\n" % (len(self.words),sum(len(s) for s in self.words))
                    self.set_output(message, "maroon")

            if not self.exit_flag:
                self.pointer = self.OutputText.GetInsertionPoint()
                self.display_sent = []
                self.max_tags = len(self.words[self.sent_num])
                self.tag_sent = self.tags[self.sent_num]#['0' for col in range(self.max_tags)]
                for jj in range(self.max_tags):
                    self.display_sent.append(self.words[self.sent_num][jj]+'('+str(jj)+')')
                message = '\n\nSentence{}: {}\n'.format(self.sent_num, ' '.join(self.display_sent))
                message += 'Current tags: %s\n'%' '.join(self.tags[self.sent_num])
                self.set_output(message, "blue")
                self.OutputText.AppendText("Input the total actions and their indices:\n")
                
        self.InputText.Clear()
            
            
    def Search_Button(self, event):
        cur.execute('select * from '+self.table)#选择table tag_actions
        self.result = cur.fetchall()
        self.OutputText.AppendText("\nTotal records: %d"%len(self.result))
        self.OutputText.AppendText("\nHow many records do you want to display:\
            \nA for all records \n L for last ten records \n D for delete last record\n ")
        self.search_flag = True
           

    def ShowAll_Button(self, event):
        self.OutputText.AppendText(
            "\n-------------------------All Records In The Database-------------------------\n")
        self.OutputText.AppendText("\nTotal records: %d"%len(self.result))
        for detail in self.result:
            self.OutputText.AppendText('\n'+str(detail))
        self.OutputText.AppendText("\nTotal records: %d"%len(self.result))
        self.OutputText.AppendText(
            "\n-------------------------------End Of Display--------------------------------\n")
        self.search_flag = False
        
        if self.start_flag:
            message = '\n\nSentence{}: {}\n'.format(self.sent_num, ' '.join(self.display_sent))
            self.set_output(message, "blue")
            self.OutputText.AppendText("Input the indexs and amount of actions in this sentence:\n")
        else:
            self.OutputText.AppendText("Input text_num and sent_num:\n")
    

    def ShowLast10_Button(self, event):
        self.OutputText.AppendText(
            "\n----------------------------Last Ten Records-------------------------------\n")
        if len(self.result) >= 10:
            start = len(self.result)-10
        else:
            start = 0
        for di in range(start,len(self.result)):
            self.OutputText.AppendText('\n'+str(self.result[di]))
        self.OutputText.AppendText(
            "\n-----------------------------End Of Display--------------------------------\n")
        self.search_flag = False
        
        if self.start_flag:
            message = '\n\nSentence{}: {}\n'.format(self.sent_num, ' '.join(self.display_sent))
            self.set_output(message, "blue")
            self.OutputText.AppendText("Input the indexs and amount of actions in this sentence:\n")
        else:
            self.OutputText.AppendText("Input text_num and sent_num:\n")
    

    def Delete_Button(self, event):
        self.delete_flag = True
        message = "\nPlease input text_num and sent_num in the Search input box:\n"
        self.set_output(message, "red")
    

    def Search_Input(self, event):
        if self.delete_flag:
            s = self.SearchText.GetValue().encode("ascii")
            temp_in = s.split()
            if len(temp_in)!=2:
                message = '\nText_num out of range. Please input again:\n'
                self.set_output(message, "red")
                self.SearchText.Clear()
            else:
                tn = temp_in[0]
                sn = temp_in[1]
                delete = "delete from "+self.table+" where text_num="+tn+" and sent_num="+sn
                self.OutputText.AppendText('\ndelete'+delete+'\n')
                cur.execute(delete)
                db.commit()
                cur.execute('select * from '+self.table)#选择table tag_actions
                result = cur.fetchall()
                self.OutputText.AppendText(
                    "\n----------------------------Last Ten Records-------------------------------\n")
                if len(result) >= 10:
                    start = len(result)-10
                else:
                    start = 0
                for di in range(start,len(result)):
                    self.OutputText.AppendText('\n'+str(result[di]))
                self.OutputText.AppendText(
                    "\n-----------------------------End Of Display--------------------------------\n")
                
                self.delete_flag = False
                self.start_flag = False
                self.SearchText.Clear()
                self.OutputText.AppendText("Input text_num and sent_num:\n")
                
        elif self.change_flag:
            s = self.SearchText.GetValue().encode("ascii")
            self.home_dir = s
            self.OutputText.AppendText('\nYour input: %s\n'%s)
            self.OutputText.AppendText(
                '\nPlease input a new table_name and text_num in Search input box\n')
            self.change_flag = False
            self.SearchText.Clear()
        
        elif self.table_flag:
            s = self.SearchText.GetValue().encode("ascii").split()
            if len(s) != 2:
                self.OutputText.AppendText('\nWrong input!! Please input again.\n')
                self.SearchText.Clear()
            else:
                if s[0][0:-1] != 'tag_actions':
                    self.OutputText.AppendText('\nWrong table name!! Please input again.\n')
                    self.SearchText.Clear()
                else:
                    self.table = s[0]
                    self.total_text = int(s[1])
                    self.SearchText.Clear()
                    self.OutputText.AppendText('\nNew table_name: %s  total_text: %s\n'%(s[0],s[1]))
                    self.table_flag = False
                    self.text_num = 0
                    self.sent_num = 0
                    self.num = 0
                    self.max_tags = 0
                    self.words = []
                    self.tag_sent = []
                    self.display_sent = []
                    self.start_flag = False
                    self.search_flag = False
                    self.delete_flag = False
                    self.OutputText.SetValue("*****System reboot finishd.*****\n")
                    message = "Input text_num and sent_num:\n"
                    self.set_output(message, "red")
        
        else:
            self.SearchText.Clear()
            message = "\nYou should not input here now!\n"
            self.set_output(message, "red")

    def Change_dir_Button(self, event):
        self.OutputText.AppendText('\nCurrent home_dir is %s\n' % self.home_dir)
        self.OutputText.AppendText('\nPlease input a new home_dir in Search input box\n')
        self.change_flag = True
        self.table_flag = True

    def Exit_Button(self, event):
        print '\nText_num: %d; Sentence_num: %d' %(self.text_num,self.sent_num)
        end_time = time.time()
        self.OutputText.AppendText('\nText_num: %d; Sentence_num: %d' %(self.text_num,self.sent_num))
        self.OutputText.AppendText('Total time cost: %ds'%(end_time-start_time))
        print 'Total time cost: %ds'%(end_time-start_time)
        print '\nCurrent time is %s\n'%time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        wx.Exit()

    def Reboot_Button(self, event):
        self.text_num = 0
        self.sent_num = 0
        self.num = 0
        self.max_tags = 0
        self.words = []
        self.tag_sent = []
        self.display_sent = []
        self.start_flag = False
        self.search_flag = False
        self.delete_flag = False
        self.OutputText.SetValue("*****System reboot finishd.*****\n")
        message = "Input text_num and sent_num:\n"
        self.set_output(message, "red")
        
        
if __name__ == '__main__':
    #import sys
    #reload(sys)
    #sys.setdefaultencoding('gb18030')
    #import pdb
    #pdb.set_trace()
    start_time = time.time()
    db = mysql.connector.connect(user='fengwf',password='123',database='test')
    cur = db.cursor()
    app = wx.PySimpleApp()
    frame = TextFrame()
    frame.Show()
    app.MainLoop()