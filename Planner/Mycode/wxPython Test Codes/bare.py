#coding:utf-8
"""Hello, wxPython! program."""
import wx

class Frame(wx.Frame): #2 wx.Frame����
    """Frame class that displays an image."""
    def __init__(self, image, parent=None, id=-1, pos=wx.DefaultPosition, title='Hello, wxPython'): #3ͼ�����
        """Create a Frame instance and display image."""
        #4 ��ʾͼ��
        temp = image.ConvertToBitmap()
        size = temp.GetWidth(), temp.GetHeight()
        wx.Frame.__init__(self, parent, id, title, pos, size)
        self.bmp = wx.StaticBitmap(parent=self, bitmap=temp)
    
    
class App(wx.App): #5 wx.App����
    """Application class."""
    def OnInit(self):
        #6 ͼ����
        image = wx.Image('kn.jpg', wx.BITMAP_TYPE_JPEG)
        self.frame = Frame(image)
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True
        
        

app = App()
app.MainLoop()
    
 