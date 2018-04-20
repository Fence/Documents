#coding:utf-8
import wx

class TextFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'Text Entry Example', size=(500, 400))
        panel = wx.Panel(self, -1)
        static = wx.StaticText(panel, -1, "Multi-line text\ncan also\n"\
        "be right aligned\n\neven with a blank",style=wx.ALIGN_RIGHT)
        str = "You can also change the font."
        text = wx.StaticText(panel, -1, str)
        font = wx.Font(18, wx.DECORATIVE, wx.ITALIC, wx.NORMAL)
        text.SetFont(font)
        
        
        self.basicLabel = wx.StaticText(panel, -1, "Basic Control:")
        self.basicText = wx.TextCtrl(panel, -1, "I've entered some text!", size=(175, -1))
        self.basicText.SetInsertionPoint(0)
        self.basicLabel2 = wx.StaticText(panel, -1, "Basic Control2:")
        self.basicText2 = wx.TextCtrl(panel, -1, "abc", size=(175, -1))#, style=wx.TE_PROCESS_ENTER)
        self.basicText2.SetInsertionPoint(0)
        self.Bind(wx.EVT_TEXT_ENTER,self.OnText,self.basicText2)
        #help(basicText)
        self.pwdLabel = wx.StaticText(panel, -1, "Password:")
        self.pwdText = wx.TextCtrl(panel, -1, "password", size=(175, -1), style=wx.TE_PASSWORD)
        self.sizer = wx.FlexGridSizer(cols=2, hgap=10, vgap=30)
        self.sizer.AddMany([static,text,\
        self.basicLabel, self.basicText, self.basicLabel2, self.basicText2, \
        self.pwdLabel, self.pwdText])
        panel.SetSizer(self.sizer)
        self.button = wx.Button(panel, -1,"Hello", pos=(50, 300))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.button)
        self.button.SetDefault()
        
    def OnText(self, event):
        self.basicText.SetValue("What the fuck.")
        
    def OnClick(self, event):
        self.button.SetLabel("Clicked")
        #self.basicText.Clear()
        #self.basicText.SetValue("What the fuck.")
        s = self.basicText.GetValue()
        #print len(s)
        s = s.encode("ascii") 
        for i in s:
                #if int(i) == 1:
                    #print 'i==1'
            print i,type(i)
        self.basicText2.SetValue(s)
        
if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = TextFrame()
    frame.Show()
    app.MainLoop()