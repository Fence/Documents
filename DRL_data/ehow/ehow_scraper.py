import re
import os
import time
import pickle
import requests
from lxml import etree

home = 'https://www.ehow.com'
categories = ['/home','/crafts','/food','/parties','/fashion',
        '/holidays','/garden','/personal-finance','/pets']

def get_urls():
    urls = {}
    for c in categories:
        urls[c] = []
    for cate in categories:
        cate_url = home + cate

        page = requests.get(cate_url).content
        tree = etree.HTML(page)
        ahrefs = tree.xpath('//div[@class="tile-title"]/a[@href]')
        print('\nAdding urls for category %s'%cate)
        for a in ahrefs:
            url = a.get('href')
            if url.endswith('html'):
                urls[cate].append(url)
                print('Real url: %s'%url)
            else:
                print('Wrong url: %s'%url)

        # if 'load more' button exists, load it
        but = tree.xpath('//button[@class="load_more_btn "]')
        if but:
            print('\nLoad more button exists. Loading more urls...\n')
            print('New page is: %s\n'%('https:' + but[0].get('data-url')))
            page2 = requests.get('https:' + but[0].get('data-url')).content
            tree2 = etree.HTML(page2)
            ahrefs2 = tree2.xpath('//@data-url')
            for a in ahrefs2:
                url = a.split('/')[1]
                if url.endswith('html'):
                    urls[cate].append(url)
                    print('Real url: %s'%url)
                else:
                    print('Wrong url: %s'%url)

    for k in urls:
        print('Total pages of %s: %d'%(k, len(urls[k]))) 

    return urls

def scrap_a_page(url):
    page = requests.get(home+url).content
    tree = etree.HTML(page)

    info = steps = []
    file_name = 'raw_data/'+url[1:-5]+'.txt'
    with open(file_name, 'w') as f:
        info = tree.xpath('//section[@class="subsection"]/span/span/p')
        if info:
            print('Introduction:\t%s'%info[0].text)
            f.write(info[0].text+'\n')
        steps = tree.xpath('//div[@class="step"]/span/span/p')
        if steps:
            for i,s in enumerate(steps):
                print('Steps %d:%s'%(i,s.text))
                f.write(s.text+'\n')

    if not steps:
        os.system('rm %s'%file_name)
        print('\n-----File %s is removed.-----\n'%file_name)
        time.sleep(3)


if __name__ == '__main__':
    start = time.time()
    if os.path.exists('urls.pkl'):
        urls = pickle.load(open('urls.pkl', 'rb'))
    else:
        with open('urls.pkl', 'wb') as f:
            urls = get_urls()
            pickle.dump(urls, f)  
    totalpage = sum([len(urls[u]) for u in urls.keys()])
    count = 0
    for cate in urls.keys():
        print('\n\nProcessing category %s\n\n'%cate)
        for url in urls[cate]:
            try:
                print('\n\nProcessing page %d of %d'%(count, totalpage))
                print('url: %s\n'%(home+url))
                count += 1
                scrap_a_page(url)
            except Exception as e:
                print('An error raised:',e)
    end = time.time()
    print('\nTotal text: %d\nTotal time cost: %d\n'%(totalpage, end-start))
