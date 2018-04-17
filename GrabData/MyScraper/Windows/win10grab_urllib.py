import urllib 
import urllib2 
url = 'https://support.microsoft.com/en-us/help/10164/fix-windows-update-errors' 
#user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)' 
  
values = {'name' : 'Michael Foord', 
          'location' : 'microsoft', 
          'language' : 'Python' } 
#headers = { 'User-Agent' : user_agent } 
#data = urllib.urlencode(values) 
#req = urllib2.Request(url, data, headers) 
#response = urllib2.urlopen(req) 
#request = urllib2.Request(url)
#request.add_header('User-Agent', 'fake-client')
response = urllib2.urlopen(url)
the_page = response.read()
f = open("the_page3.txt",'w+')
f.write(the_page)
f.close()



'''import re
import time
import urllib

url = "https://support.microsoft.com/en-us/help/12386/windows-10-find-lost-files-after-update"
f = open("pure_text.txt",'w+')
a = urllib.urlopen(url).read()
b = re.sub(r'<.+>',' ',a)
f.write(b)
print b
print "done"
f.close()'''