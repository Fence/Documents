#coding:utf-8
from selenium import webdriver

browser = webdriver.Chrome('./chromedriver')
url = 'https://wx2.qq.com/'
browser.get(url)
input_box = browser.find_elements_by_id('editArea')[0]
times = 10
interval = 1
for i in range(times):
    input_box.send_keys('testing')
    browser.find_element_by_xpath("//a[contains(text(),'发送')]").click()
    time.sleep(interval)
browser.quit()
