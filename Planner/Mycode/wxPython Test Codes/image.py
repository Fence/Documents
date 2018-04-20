#coding:utf-8
import wx
filenames = ["1.jpg", "2.jpg", "3.jpg", "kn.jpg" ]
class TestFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="Loading Images")
        self.panel = wx.Panel(self)  
        self.panel.Bind(wx.EVT_ERASE_BACKGROUND,self.OnEraseBack) 
        '''
        panel = wx.Panel(self)
        name = '2.jpg'
        img1 = wx.Image(name, wx.BITMAP_TYPE_ANY)
        w = img1.GetWidth()
        h = img1.GetHeight()
        print w,h
        img2 = img1.Scale(w/4, h/4).ConvertToBitmap()  #2 缩小图像
        self.background = wx.StaticBitmap(panel, -1, img2, (0,0))
        '''
        '''
        fgs = wx.FlexGridSizer(cols=2, hgap=10, vgap=10)
        for name in filenames:
            #1 从文件载入图像
            img1 = wx.Image(name, wx.BITMAP_TYPE_ANY)
            # Scale the original to another wx.Image
            w = img1.GetWidth()
            h = img1.GetHeight()
            img2 = img1.Scale(w/8, h/8)#2 缩小图像
            #3 转换它们为静态位图部件
            #sb1 = wx.StaticBitmap(panel, -1, wx.BitmapFromImage(img1))
            sb2 = wx.StaticBitmap(panel, -1, wx.BitmapFromImage(img2))
            # and put them into the sizer
            #fgs.Add(sb1)
            fgs.Add(sb2)
        panel.SetSizerAndFit(fgs)
        '''
        #self.Fit()
    def OnEraseBack(self,event):  
        dc = event.GetDC()  
        if not dc:  
            dc = wx.ClientDC(self)  
            rect = self.GetUpdateRegion().GetBox()  
            dc.SetClippingRect(rect)  
        dc.Clear()  
        bmp = wx.Bitmap("2.jpg")  
        dc.DrawBitmap(bmp, 0, 0)  
    
app = wx.PySimpleApp()
frm = TestFrame()
frm.Show()
app.MainLoop()