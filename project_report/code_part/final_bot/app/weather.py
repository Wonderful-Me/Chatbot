

from bs4 import BeautifulSoup
import urllib
from xpinyin import Pinyin

def get_weather(sentence):
    p=Pinyin()
    #将输入的汉字用Pinyin转化为拼音字符串
    city_pinyin=p.get_pinyin(sentence,'')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/51.0.2704.63 Safari/537.36'}
    try:
        url = 'https://www.tianqi.com/'+city_pinyin
        r = urllib.request.Request(url=url, headers=headers)
    except urllib.error.URLError as e:
        return "抱歉哈，找不到你所输入的城市"

    r = urllib.request.urlopen(r)
    info_weather = r.read().decode("utf-8")
    soup = BeautifulSoup(info_weather, "html.parser")
    weather=[]
    city_find=soup.select('.weatherbox .wrap1100 .left .weather_info .name h1')[0].text
    pinyin=p.get_pinyin(city_find,'')
    pinyin2=p.get_pinyin('天气','')
    if(pinyin!=city_pinyin+pinyin2):
        return "抱歉哈，找不到你所输入的城市"

    weather.append(soup.select('.weatherbox .wrap1100 .left .weather_info .name h1')[0].text+\
                   '('+soup.select('.weatherbox .wrap1100 .left .weather_info .week ')[0].text+')：')
    weather.append('当前温度：'+soup.select('.weatherbox .wrap1100 .left .weather_info .weather .now')[0].text)
    weather.append('当日天气情况：'+soup.select('.weatherbox .wrap1100 .left .weather_info .weather span ')[0].text)
    weather.append(soup.select('.weatherbox .wrap1100 .left .weather_info .shidu b')[0].text)
    weather.append(soup.select('.weatherbox .wrap1100 .left .weather_info .shidu b')[1].text)
    weather.append(soup.select('.weatherbox .wrap1100 .left .weather_info .shidu b')[2].text)
    weather.append(soup.select('.weatherbox .wrap1100 .left .weather_info .kongqi h5 ')[0].text)
    for i in range(0,7):
        #weather[i].strip()
        return weather[i]

    return 1

if __name__ == '__main__':
    while(1):
        if(get_weather()==1):
            break
