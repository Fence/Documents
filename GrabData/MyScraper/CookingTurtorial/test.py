# -*- coding: utf-8 -*-  
import re
import time 
from bs4 import BeautifulSoup as bs
from selenium import webdriver  
from selenium.common.exceptions import NoSuchElementException  
  

class GrabData():
    def __init__(self, browser):  
        print "\n\nInitializing GrabDta...\n"
        self.browser = browser
        self.home = "http://cookingtutorials.com/"
        self.category_url = []
        self.category_name = []
        self.subpage_name = []
        self.sub_url = []
        self.totalpages = 23

    def grab_a_page(self, page_name, url):
        print '\nGrabbing data from "%s"'%url
        self.browser.get(url)
        self.browser.implicitly_wait(5)

        f = open('raw_texts/' + page_name + '.txt','w+')
        soup = bs(self.browser.page_source,'xml')
        a = soup.find_all('p',{'style':'text-align: justify;'})

        for b in a:
            c = re.sub(r'[\.\?\!\:]','\n',b.text)
            d = re.findall(r'\w+\n?',c)
            for i in d:
                if i[-1] == '\n':
                    f.write(i)
                else:
                    f.write(i+' ')
        f.close()


    def grab_single_page(self, page_name, url):
        print '\nGrabbing data from "%s"'%url
        self.browser.get(url)
        self.browser.implicitly_wait(5)

        f = open('raw_texts/' + page_name + '.txt','w+')
        a = self.browser.find_elements_by_tag_name('p')
        for ind in range(len(a)-3):
            i = a[ind]
            t = i.text
            if len(t) > 1:
                b = re.sub(r'[\.\?\!\:]','\n',t)
                c = re.findall(r'[\w\-]+\n?',b)
                for e in c:
                    if e[-1] == '\n':
                        f.write(e)
                    else:
                        f.write(e+' ')
        f.close()


    def guidepage(self, page_name, url):
        self.browser.get(url)
        self.browser.implicitly_wait(10)

        
        soup = bs(self.browser.page_source,'xml')
        temp = soup.find_all('div',{'class':'post_header half'})

        for i in temp:
            a = i.find('a')
            self.subpage_name.append(a.text)
            self.sub_url.append(a.get('href'))
            self.mainpage.write(a.text + '\n' + a.get('href') + '\n')



    def main(self):
        
        self.mainpage = open('mainpage.txt','w+')
        self.mainpage.write('categories:\n')
        self.browser.get(self.home)
        self.browser.implicitly_wait(10)
        soup = bs(self.browser.page_source,'xml')
        temp1 = soup.find_all('div',{'class':'tagcloud'})
        temp2 = temp1[0].find_all('a')
        for i in temp2:
            self.category_name.append(i.text)
            self.category_url.append(i.get('href'))
            self.mainpage.write(i.text+'\n'+i.get('href')+'\n')
        self.mainpage.write('\n\nsubpages:\n')

        total_categories = len(self.category_url)
        print "Total categories: %d"%total_categories
        for i in range(total_categories):
            print 'Opening category %s: "%s"'%(self.category_name[i], self.category_url[i])
            self.guidepage(self.category_name[i], self.category_url[i])

        '''
        for i in range(self.totalpages):
            if i == 0:
                print 'Opening page "%s"'%self.home
                self.guidepage('page_' + str(i+1), self.home)  
            else:
                print 'Opening page "%s"'%(self.home + 'page/' + str(i+1) + '/')
                self.guidepage('page_' + str(i+1), self.home + 'page/' + str(i+1) + '/')             
        
        temp = open('mainpage.txt')
        count = 0
        for line in temp.readline():
            count += 1
            if count % 2 == 1:
                self.subpage_name.append(line[:-1])
            else:
                self.sub_url.append(line[:-1])
        '''
        num_of_subpages = len(self.subpage_name)
        print "\n\nTotal subpages :%d\n"%len(self.subpage_name)
        
        for i in range(len(self.subpage_name)):
            print "----------  page %d of %d  ----------"%(i+1, num_of_subpages)
            self.grab_single_page(self.subpage_name[i], self.sub_url[i])
        self.mainpage.close()
        print "\n\nMain function is finished."







if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('gb18030')
    start = time.time()
    browser = webdriver.Chrome()
    test = GrabData(browser)
    test.main()
    a = raw_input("Quit? (Y/N)")
    if not a or a.upper() == 'Y':
        browser.quit()
    end = time.time()
    print "\n\nTotal time cost: %ds"%(end-start)