'''
how do a web try to anti-crawler:
1) limit the visist frequency of certain ip=> disconnect the connection when the connection frequency is too high, so we can try to sleep
2) block certain agent access after its visiting threshold is too high, ususally not adopted by webs due to its high error occurance
3)analysis cookies, usually not adopted by web
'''
import programmingalpha
import requests
import re
import random
import time

class AgentProxyCrawler(object):

    def __init__(self):
        self.ip_list=[]
        #html=requests.get("http://www.haoip.com")
        #print(html.text)
        #ip_pattern=r'(?=(\b|\D))(((\d{1,2})|(1\d{1,2})|(2[0-4]\d)|(25[0-5]))\.){3}((\d{1,2})|(1\d{1,2})|(2[0-4]\d)|(25[0-5]))(?=(\b|\D))'

        #iplistn=re.findall(ip_pattern,html.text,re.S)
        with open(programmingalpha.ConfigPath+"ip-proxy.txt","r") as f:
            ips=f.readlines()
        for ip in ips:
            #print(ip)
            ip=ip.strip()
            if len(ip)<7:
                continue
            i=ip.find("#")
            ip=ip[:i]
            self.ip_list.append(ip)

        self.user_agent_list=[
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
            "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
            "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
        ]
        print("init with %d ips"%len(self.ip_list))
        print(self.ip_list[:1],"...")

    def get(self,url,params,timeout,proxy=None,num_retries=6):
        ua=random.choice(self.user_agent_list)
        header={"User-Agent":ua}

        if proxy==None:
            try:
                response=requests.get(url,params=params,headers=header,timeout=timeout)
                return response
            except:
                if num_retries>0:
                    time.sleep(10)
                    print(u"failed to access web, left trail times：",num_retries-1)
                    return self.get(url,params,timeout,num_retries-1)
                else:
                    print("begin to use the proxy")
                    time.sleep(10)
                    IP="".join(str(random.choice(self.ip_list)).strip())
                    proxy={"http":IP}
                    return self.get(url,params,timeout,proxy)

        else:
            try:
                IP="".join(str(random.choice(self.ip_list)).strip())
                proxy={"http":IP}
                response=requests.get(url,params=params,headers=header,proxies=proxy,timeout=timeout)
                return response
            except:
                if num_retries>0:
                    time.sleep(10)
                    IP="".join(str(random.choice(self.ip_list)).strip())
                    print(u"trying to change the proxy ip, left trial times",num_retries-1)
                    print(u"current proxy：",proxy)
                    return self.get(url,params,timeout,proxy,num_retries-1)
                else:
                    print(u"error while using proxy, rollback!!!")
                    return self.get(url,params,3)

