# -*- coding: utf-8 -*-  
import re
import time 
from bs4 import BeautifulSoup as bs
from selenium import webdriver  
from selenium.common.exceptions import NoSuchElementException  
  
''' 
def find_sec(secid):  
    pa=re.compile(r'\w+')  
    browser = webdriver.Firefox() # Get local session of firefox  
    browser.get("http://bbs.byr.cn/#!section/%s "%secid) # Load page  
    time.sleep(1) # Let the page load  
    result=[]  
    try:  
        #获得版面名称和在线人数，形成列表  
        board=browser.find_elements_by_class_name('title_1')  
        ol_num=browser.find_elements_by_class_name('title_4')  
        max_bindex=len(board)  
        max_oindex=len(ol_num)  
        assert max_bindex==max_oindex,'index not equivalent!'  
          
        #版面名称有中英文，因此用正则过滤只剩英文的  
        for i in range(1,max_oindex):  
            board_en=pa.findall(board[i].text)  
            result.append([str(board_en[-1]),int(ol_num[i].text)])  
              
        browser.close()  
        return result  
    except NoSuchElementException:  
        assert 0, "can't find element"  
  
#打印分区5下面的所有板块的当前在线人数列表          
#print find_sec('5')  
'''
class GrabData():
    def __init__(self, browser):  
        self.browser = browser
        self.head = "https://support.microsoft.com"
        self.main_win10 = self.head + "/en-us/products/windows?os=windows-10"
        self.main_account = self.head + "/en-us/products/microsoft-account"
        self.main_lifecycle = self.head + "/en-us/lifecycle"
        self.subpage_name = []
        self.sub_url = []
        self.err_logs = open('err_logs.txt','a+')

    def grab_a_page(self, fold_name, page_name, url):
        #url = "https://support.microsoft.com/en-us/help/12386/windows-10-find-lost-files-after-update"
        print '\nGrabbing data from "%s"'%url
        self.browser.get(url)
        #time.sleep(1)
        self.browser.implicitly_wait(10)

        f = open(fold_name + page_name + '.txt','w+')
        soup = bs(self.browser.page_source,'xml')
        a = soup.find_all('div',{'class':'ng-scope'})
        if len(a) >= 10:           
            b = a[9].text
            c = re.sub(r'\n',' ',b)
            d = re.sub(r'[\.\?\!\:]','\n',c)
            e = re.findall(r'\w+\n?',d)
            for i in e:
                if i[-1] == '\n':
                    f.write(i)
                else:
                    f.write(i+' ')
            
        else:
            self.err_logs.write('Error page name: %s\nError url: %s\n\n'%(page_name,url))
            for i in range(len(a)):
                b = a[i].text
                c = re.sub(r'\n',' ',b)
                d = re.sub(r'[\.\?\!\:]','\n',c)
                e = re.findall(r'\w+\n?',d)
                for i in e:
                    if i[-1] == '\n':
                        f.write(i)
                    else:
                        f.write(i+' ')
        f.close()


    def mainpage(self, page_name, url, begin, stop):
        print "\n\n++++++++++++++++++++++++++++++++++++++++++++"
        print "\n\nGrabbing the main page:\n %s..."%page_name
        self.browser.get(url)
        self.browser.implicitly_wait(10)

        f = open(page_name + '.txt','w+')
        soup = bs(self.browser.page_source,'xml')
        temp_url = soup.find_all('a',{'class':'ng-binding'})

        count = 0
        for i in range(len(temp_url)):
            j = temp_url[i]
            if j.get('href'):
                count += 1
                if count >= begin and count <= stop:
                    f.write(j.text + '\n    ' + j.get('href') + '\n')
                    self.subpage_name.append(j.text)
                    self.sub_url.append(j.get('href'))
        f.close()
        print "\nTotal amount of subpages: %d"%len(self.subpage_name)


    def main(self):
        ###
        self.mainpage('win10_mainpage', self.main_win10, 14, 171)
        num = len(self.subpage_name)
        for i in range(num):
            print "----------  page: %d of %d  ----------"%(i+1,num)
            self.grab_a_page('./win10/', self.subpage_name[i], self.head + self.sub_url[i])
        
        ###
        self.mainpage('account_mainpage', self.main_account, 6, 79)
        num = len(self.subpage_name)
        for i in range(num):
            print "----------  page: %d of %d  ----------"%(i+1,num)
            self.grab_a_page('./account/', self.subpage_name[i], self.head + self.sub_url[i])
        
        ###
        self.mainpage('lifecycle_mainpage', self.main_lifecycle, 1, 33)
        num = len(self.subpage_name)
        for i in range(num):
            print "----------  page: %d of %d  ----------"%(i+1,num)
            self.grab_a_page('./lifecycle/', self.subpage_name[i], self.head + self.sub_url[i])

        ###    
        print "\nMain function is finished"
        self.err_logs.close()






if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('gb18030')
    start = time.time()
    browser = webdriver.Chrome()
    test = GrabData(browser)
    test.main()
    #a = raw_input("Quit? (Y/N)")
    #if not a or a.upper() == 'Y':
    #    browser.quit()
    end = time.time()
    print "\n\nTotal time cost: %ds"%(end-start)