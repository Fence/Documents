#coding:utf-8
import wx
class TextFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'Tag Actions from texts', size=(1000, 1000))  #整个界面大小
        panel = wx.Panel(self, -1)
        panel.SetBackgroundColour('aquamarine')
        panel.Refresh()
        
        #static = wx.StaticText(panel, -1, "Multi-line text\ncan also\n""be right aligned\n\neven with a blank",style=wx.ALIGN_RIGHT)
        #right = wx.StaticText(panel, -1, "align right", (500, 0), (160, -1), wx.ALIGN_RIGHT)
        str = "Welcome to the tag_actions.app from Fence!"  #静态显示文本，欢迎标题
        self.text = wx.StaticText(panel, -1, str, (140,-1))
        font1 = wx.Font(18, wx.ROMAN, wx.ITALIC, wx.BOLD)  #字体18号，罗马字，斜体，加粗
        self.text.SetFont(font1)
        self.text.SetForegroundColour('tan')  #颜色为棕色

        self.multiText = wx.TextCtrl(panel, -1,"Input text_num and sent_num:\n",
        size=(700, 400), style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2) #创建一个输出文本控件
        font2 = wx.Font(12, wx.SCRIPT, wx.NORMAL, wx.NORMAL)
        self.multiText.SetFont(font2)
        #self.multiText.SetForegroundColour('orchid')
        self.pointer = self.multiText.GetInsertionPoint()
        #print 'self.pointer',self.pointer
        
        self.basicLabel = wx.StaticText(panel, -1, "Your input:", (100,455)) #输入文本框
        self.basicText = wx.TextCtrl(panel, -1, "", size=(175, -1), \
        style=wx.TE_PROCESS_ENTER)
        self.basicText.SetInsertionPoint(0)
        self.Bind(wx.EVT_TEXT_ENTER,self.OnText,self.basicText)
        
        self.basicLabel2 = wx.StaticText(panel, -1, "Search input:", (390,455)) #输入文本框
        self.basicText2 = wx.TextCtrl(panel, -1, "", size=(175, -1), \
        style=wx.TE_PROCESS_ENTER)  #查询的输入文本框
        self.basicText2.SetInsertionPoint(0)
        self.Bind(wx.EVT_TEXT_ENTER,self.OnText2,self.basicText2)

        self.button1 = wx.Button(panel, -1,"Search")  #搜索按钮
        self.Bind(wx.EVT_BUTTON, self.Search_Button, self.button1)
        self.button1.SetDefault()
        
        self.button2 = wx.Button(panel, -1,"A")  #按钮，显示全部记录
        self.Bind(wx.EVT_BUTTON, self.ShowAll_Button, self.button2)
        self.button2.SetDefault()
        
        self.button3 = wx.Button(panel, -1,"L")  #按钮，显示最近十条记录
        self.Bind(wx.EVT_BUTTON, self.ShowLast10_Button, self.button3)
        self.button3.SetDefault()
        
        self.button4 = wx.Button(panel, -1,"D")  #按钮，删除记录
        self.Bind(wx.EVT_BUTTON, self.Delete_Button, self.button4)
        self.button4.SetDefault()
        
        self.button5 = wx.Button(panel, -1,"Exit")  #按钮，退出
        self.Bind(wx.EVT_BUTTON, self.Exit_Button, self.button5)
        self.button5.SetDefault()
        
        self.button6 = wx.Button(panel, -1,"Reboot")  #按钮，退出
        self.Bind(wx.EVT_BUTTON, self.Reboot_Button, self.button6)
        self.button6.SetDefault()
        
        sizer1 = wx.FlexGridSizer(cols=1, hgap=50, vgap=50)
        sizer1.Add(self.text,1,wx.EXPAND)
        sizer1.Add(self.multiText,1,wx.EXPAND)
        #sizer1.AddMany([self.text,self.multiText])
        sizer2 = wx.FlexGridSizer(cols=4, hgap=50, vgap=50)
        sizer2.Add(self.basicLabel,1,wx.EXPAND)
        sizer2.Add(self.basicText,1,wx.EXPAND)
        sizer2.Add(self.basicLabel2,1,wx.EXPAND)
        sizer2.Add(self.basicText2,1,wx.EXPAND)
        #sizer2.AddMany([self.basicLabel,self.basicText,self.basicLabel2,self.basicText2])
        sizer3 = wx.FlexGridSizer(cols=6, hgap=10, vgap=50)
        sizer3.Add(self.button1,1,wx.EXPAND)
        sizer3.Add(self.button2,1,wx.EXPAND)
        sizer3.Add(self.button3,1,wx.EXPAND)
        sizer3.Add(self.button4,1,wx.EXPAND)
        sizer3.Add(self.button5,1,wx.EXPAND)
        sizer3.Add(self.button6,1,wx.EXPAND)
        #sizer3.AddMany([self.button1,self.button2,self.button3,self.button4,self.button5,self.button6])
        sizer4 = wx.FlexGridSizer(cols=1, hgap=50, vgap=50)
        sizer4.AddMany([sizer1,sizer2,sizer3])
        panel.SetSizer(sizer4)
        
    def OnText(self, event):
        pass
    def Search_Button(self, event):
        pass
    def ShowAll_Button(self, event):
        pass
    def ShowLast10_Button(self, event):
        pass
    def Delete_Button(self, event):
        pass
    def OnText2(self, event):
        pass
    def Exit_Button(self, event):
        pass
    def Reboot_Button(self, event):
        pass
        
        
        
        
        
            
    '''
        wx.Frame.__init__(self, None, -1, 'Text Entry Example', size=(300, 250))
        panel = wx.Panel(self, -1)
        multiLabel = wx.StaticText(panel, -1, "Multi-line")
        multiText = wx.TextCtrl(panel, -1,
        "Here is a looooooooooooooong line of text set in the control.\n\n"
        "See that it wrapped, and that this line is after a blank",
        size=(200, 100), style=wx.TE_MULTILINE) #创建一个文本控件
        multiText.SetInsertionPoint(0) #设置插入点
        
        richLabel = wx.StaticText(panel, -1, "Rich Text")
        richText = wx.TextCtrl(panel, -1,
        "If supported by the native control, this is reversed, and this is a different font.",
        size=(200, 100), style=wx.TE_MULTILINE|wx.TE_RICH2) #创建丰富文本控件
        richText.SetInsertionPoint(0)
        
        richText.SetStyle(44, 52, wx.TextAttr("white", "black")) #设置文本样式
        points = richText.GetFont().GetPointSize()
        f = wx.Font(points + 3, wx.ROMAN, wx.ITALIC, wx.BOLD, True) #创建一个字体
        richText.SetStyle(68, 82, wx.TextAttr("blue", wx.NullColour, f)) #用新字体设置样式
        
        sizer = wx.FlexGridSizer(cols=2, hgap=6, vgap=6)
        sizer.AddMany([multiLabel, multiText, richLabel, richText])
        panel.SetSizer(sizer)
    '''
if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = TextFrame()
    frame.Show()
    app.MainLoop()