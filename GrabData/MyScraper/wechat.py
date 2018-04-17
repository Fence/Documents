#coding:utf-8
import os
import time
from selenium import webdriver

class wechat():
    def __init__(self):
        self.browser = webdriver.Chrome('./chromedriver')
        self.url = 'https://wx2.qq.com/'
        self.confirm = 'unset'
        self.browser.get(self.url)
        self.confirm = raw_input("Please scan the QR code to log in your wechat accuont.\n\
        Then open a dialog with your friend(s) and input 'Y' or ENTER to continue.\n")
        self.browser.implicitly_wait(3)
        self.log_in()


    def log_in(self):
        while (self.confirm and self.confirm.upper() != 'Y'):  
            self.confirm = raw_input("Please scan the QR code to log in your wechat accuont.\n\
            Then open a dialog with your friend(s) and input 'Y' or ENTER to continue.\n")


    def choose_input_type(self):
        choice = raw_input('Send from files or by keybord? (f or k)\n').lower()
        while choice not in ['f','k']:
            choice = raw_input('Send from files or by keybord? (f or k)\n').lower()
        if choice == 'f':
            file_name = raw_input('Input the file name:\n')
            test.auto_send_file(file_name)
        else:
            test.auto_send()

    def auto_send_file(self, file_name):
        if os.path.isfile(file_name):
            find_input = self.browser.find_elements_by_id('editArea')
            input_box = find_input[0]
            temp_time = raw_input("Input the time interval between each send.\n")
            if temp_time:
                t = float(temp_time)
            for line in open(file_name):
                if len(line) <= 2:
                    continue
                input_box.send_keys(line.decode('utf-8')) #PS：在发送前解码，而不是输入时
                #current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                #input_box.send_keys(' '+current_time)
                self.browser.find_element_by_xpath("//a[contains(text(),'发送')]").click()
                time.sleep(t)
        else:
            print 'no such file named %s'%file_name


    def auto_send(self):
        message = 'unset'
        find_input = self.browser.find_elements_by_id('editArea')
        input_box = find_input[0]

        temp_num = raw_input("How many messages do you want to send?\n")
        if temp_num:
            num = int(temp_num)
            temp_time = raw_input("Input the time interval between each send.\n")
            if temp_time:
                t = float(temp_time)
        message = raw_input("What do you want to send?\n")

        for i in range(num):
            input_box.send_keys(message.decode('utf-8')) #PS：在发送前解码，而不是输入时
            #current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            #input_box.send_keys(' '+current_time)
            self.browser.find_element_by_xpath("//a[contains(text(),'发送')]").click()
            time.sleep(t)

    def exit(self):
        print 'Bye ~ \n'
        self.browser.quit()


if __name__ == '__main__':
    test = wechat()
    test.choose_input_type()
    again = raw_input('Auto send is done, do you want to continue? (y/n)')
    while again.lower() != 'n':
        test.choose_input_type()
        again = raw_input('Auto send is done, do you want to continue? (y/n)')
    test.exit()



    