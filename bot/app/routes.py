from operator import index
import torch
import os
import argparse
from datetime import datetime
import logging
from transformers import GPT2LMHeadModel
from transformers import BertTokenizerFast
import torch.nn.functional as F
import sys
import requests
import pandas as pd
#import match
#import get_novel
#import pick_movie
#import weather

import re

def format_mark(express):
    # distinguish and replace the unexpected character that may disturb the calculation
    express = express.replace('+-', '-')
    express = express.replace('-+', '-')
    express = express.replace('++', '+')
    express = express.replace('--', '+')
    express = express.replace('*+', '*')
    express = express.replace('+*', '*')
    express = express.replace('+/', '/')
    express = express.replace('/+', '/')
    return express

def com_cal_plus_minus(express):
    expr = express
    sub_expr = re.search(r"\-?\d+\.?\d*[\+\-]\d+\.?\d*", expr)
    if not sub_expr:
        return expr
    else:
        sub_expr2 = sub_expr.group()
        if len(sub_expr2.split('+')) > 1:
            n1, n2 = sub_expr2.split('+')
            result = float(n1)+float(n2)
        else:
            n1, n2 = sub_expr2.split('-')
            result = float(n1) - float(n2)
        re_sub_expr = re.sub(r"\-?\d+\.?\d*[\+\-]\d+\.?\d*", str(result), expr, count=1)
        # calculate dicide over and over and over again
        bb = com_cal_plus_minus(str(re_sub_expr))
        return bb

def com_cal_multilply_divide(expr_div):
    expr=expr_div
    sub_expr = re.search(r"\d+\.?\d*[\/\*]\-?\d+\.?\d*",expr)
    if not sub_expr:
        return expr
    else:
        sub_expr2 = sub_expr.group()
        if len(sub_expr2.split('/')) > 1:
            n1, n2 = sub_expr2.split('/')
            result = float(n1)/float(n2)
        if len(sub_expr2.split('*')) > 1:
            n1, n2 = sub_expr2.split('*')
            result = float(n1)*float(n2)
        else:
            pass
        re_sub_expr=re.sub(r"\d+\.?\d*[\/\*]\-?\d+\.?\d*",str(result),expr,count=1)
        # calculate dicide over and over and over again
        bb=com_cal_multilply_divide(format_mark(re_sub_expr))
        return bb

def compute(express):
    express = com_cal_multilply_divide(format_mark(express))
    express = com_cal_plus_minus(format_mark(express))
    return express

def clear(express):
    # detect the blank space
    res=re.compile(r'[()]')
    sub_expr1 = re.search('(\([\+\-\*\/\.0-9]+\))', express)
    if not sub_expr1:
        return express
    else:
        sub_expr1=sub_expr1.group()
        sub_expr2=sub_expr1[1:len(sub_expr1)-1]
        sub_expr3=compute(sub_expr2)
        sub_expr3 = re.sub('(\([\+\-\*\/\.0-9]+\))', str(sub_expr3),express,count=1)
        clear_expr=clear(format_mark(sub_expr3))
        return clear_expr

def cal(express):
    while True:
        res = ""
        express = re.sub('\s*', '', express)
        if len(express) == 0:
            continue
        elif express == 'q':
            res = "取消计算啦！"
            return res
        elif re.search('[^0-9\.\-\+\*\/\(\)]',express):
            res = "不是有效的算数表达式哦，已退出计算!"
            return res
        else:
            express = express.replace(' ', '')
            express = clear(express)
            # clear the blank between discriptions
            express = compute(format_mark(express))
            # calculate again
            # in case the unexpected character to disturb the calculation
            return str(express)



import json
import random
novels = json.loads(open('F:\\USTC\\chatbot_project\\Attempt\\atm\\bot\\app\\novel.json', encoding='utf-8').read())

def get_covid(province):
    global chengshi
    global shuju
    print(province)
   
    url = 'https://api.inews.qq.com/newsqa/v1/query/inner/publish/modules/list?modules=statisGradeCityDetail,diseaseh5Shelf'
    r = requests.get(url)
    retext = r.text

    data = json.loads(retext)

    citylist = data['data']['diseaseh5Shelf']['areaTree'][0]['children']
    alist=[]
    clist=[]
    for i in citylist:

        if (i['today']['wzz_add'] == 0):
            #print(i['name'], i['today']['confirm'], 0)
            print(0, end=',')
            dic={
                'name':i['name'],
                'con':i['today']['confirm'],
                'wzz':0
            }
            clist.append([i['name'],i['today']['confirm'],0])
            alist.append(dic)
            continue
        dic = {
            'name': i['name'],
            'con': i['today']['confirm'],
            'wzz': i['today']['wzz_add']
        }
        clist.append([i['name'], i['today']['confirm'], i['today']['wzz_add']])
        print(i['today']['confirm'], end=',')
        alist.append(dic)
        #print(i['name'],i['today']['confirm'],i['today']['wzz_add'])
    #print(alist)
    pf = pd.DataFrame(list(alist))
    order = ['name', 'con', 'wzz']
    pf = pf[order]
    columns_map = {
        'name': '城市',
        'con': '新增确诊',
        'wzz': '新增无症状',
    }
    print(province)
    print(clist)
    chengshi=[]
    shuju=[]
    for i in range(1, 6):
        chengshi.append(clist[i][0])
        shuju.append(clist[i][1])

    for e in clist:
        if e[0] == province:
            return e[0] + "今日新增病例" + str(e[1]) + "人，" + "无症状感染者" + str(e[2]) + "人。\n"


def get_hotsearch():

    data = []
    response = requests.get("https://weibo.com/ajax/side/hotSearch")
    data_json = response.json()['data']['realtime']

    for data_item in data_json:
        dic = {
            'title': data_item['note'],
            'url': 'https://s.weibo.com/weibo?q=%23' + data_item['word'] + '%23',
            'num': data_item['num']
        }
        return '<a href=\"https://s.weibo.com/weibo?q=%23' + data_item['word'] + '%23\" target=\"_blank\">'+data_item['note']+'</a>'


def get_novel():
    novel_list = novels["novels"]
    result = random.choice(novel_list)
    return result

movies = json.loads(open('F:\\USTC\\chatbot_project\\Attempt\\atm\\bot\\app\\movies.json', encoding='utf-8').read())


def pick_movie(type):
    list_of_movies = movies["movies"]
    for data in list_of_movies:
        if data["tag"] == type:
            result = random.choice(data['list'])
            return result
    result = "抱歉哈，没有找到这个类型诶"
    return result

from bs4 import BeautifulSoup
import urllib
from xpinyin import Pinyin

def get_weather(sentence):
    print("weather " + sentence)
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
    res = ""
    for i in range(0, 7):
        #weather[i].strip()
        res = res + weather[i] + "    "
    print(res)
    return res





import jieba
from fuzzywuzzy import fuzz

intents = json.loads(open('F:\\USTC\\chatbot_project\\Attempt\\atm\\bot\\app\\answers.json', encoding='utf-8').read())


def is_gender(seg_list, sentence, intents):
    list_of_intents = intents['intents']
    questions = list_of_intents[0]['questions']
    for q in questions:
        if fuzz.partial_ratio(q, sentence) >= 80:
            return True
    flag_you = 0
    flag_gender = 0
    for v in seg_list:
        if v == "男" or v == "女" or v == "性别":
            flag_gender = 1
        elif v == "你" or v == "您":
            flag_you = 1
    if flag_gender == 1 and flag_you == 1:
        return True
    else:
        return False


def is_age(seg_list, sentence, intents):
    list_of_intents = intents['intents']
    questions = list_of_intents[1]['questions']
    for q in questions:
        if fuzz.partial_ratio(q, sentence) >= 80:
            return True
    flag_you = 0
    flag_age = 0
    for v in seg_list:
        if v == "年龄" or v == "几岁":
            flag_age = 1
        elif v == "你" or v == "您":
            flag_you = 1
    if flag_age == 1 and flag_you == 1:
        return True
    else:
        return False


def is_name(seg_list, sentence, intents):
    list_of_intents = intents['intents']
    questions = list_of_intents[2]['questions']
    for q in questions:
        if fuzz.partial_ratio(q, sentence) >= 80:
            return True
    flag_you = 0
    flag_name = 0
    for v in seg_list:
        if v == "姓名" or v == "名字" or v == "谁":
            flag_name = 1
        elif v == "你" or v == "您":
            flag_you = 1
    if flag_name == 1 and flag_you == 1:
        return True
    else:
        return False


def is_address(seg_list, sentence, intents):
    list_of_intents = intents['intents']
    questions = list_of_intents[3]['questions']
    for q in questions:
        if fuzz.partial_ratio(q, sentence) >= 80:
            return True
    flag_you = 0
    flag_address = 0
    for v in seg_list:
        if v == "哪里" or v == "地方":
            flag_address = 1
        elif v == "你" or v == "您":
            flag_you = 1
    if flag_address == 1 and flag_you == 1:
        return True
    else:
        return False

def Is_movie(sentence, intents):
    #list_of_intents = intents['intents']
    #questions = list_of_intents[4]['questions']
    #for q in questions:
     #   if fuzz.partial_ratio(q, sentence) >= 80:
    #        return True
    flag_question = 0
    flag_movie = 0
    seg_list = jieba.cut(sentence, cut_all=True)
    for v in seg_list:
        if v == "电影" or v == "影片":
            flag_movie = 1
        elif v == "吗" or v == "什么" or v == "嘛" or v == "推荐":
            flag_question = 1
    if flag_movie == 1 and flag_question == 1:
        return True
    else:
        return False


def Is_novel(sentence, intents):
    list_of_intents = intents['intents']
    questions = list_of_intents[5]['questions']
    for q in questions:
        if fuzz.partial_ratio(q, sentence) >= 80:
            return True
    flag_question = 0
    flag_novel = 0
    seg_list = jieba.cut(sentence, cut_all=True)
    for v in seg_list:
        if v == "小说" or v == "书":
            flag_music = 1
        elif v == "吗" or v == "什么" or v == "嘛":
            flag_question = 1
    if flag_novel == 1 and flag_question == 1:
        return True
    else:
        return False


def Is_weather(sentence, intents):
    today = 0
    list_of_intents = intents['intents']
    questions = list_of_intents[6]['questions']
    for q in questions:
        if fuzz.partial_ratio(q, sentence) >= 80:
            return True
    flag_question = 0
    flag_weather = 0
    seg_list = jieba.cut(sentence, cut_all=True)
    for v in seg_list:
        if v == "天气":
            flag_weather = 1
        elif v == "吗" or v == "什么" or v == "嘛" or v == "怎么样":
            flag_question = 1
    if flag_weather == 1 and flag_question == 1:
        return True
    else:
        return False

def Is_HE(sentence, intents):
    today = 0
    list_of_intents = intents['intents']
    questions = list_of_intents[7]['questions']
    for q in questions:
        if fuzz.partial_ratio(q, sentence) >= 80:
            return True
    return False

def Is_YQ(sentence, intents):
    today = 0
    list_of_intents = intents['intents']
    questions = list_of_intents[8]['questions']
    for q in questions:
        if fuzz.partial_ratio(q, sentence) >= 80:
            return True
    return False

def get_type(sentence, intents):
    seg_list = jieba.cut(sentence, cut_all=True)
    my_list = []
    for v in seg_list:
        my_list.append(v)
    if is_gender(my_list, sentence, intents):
        result = "gender"
    elif is_age(my_list, sentence, intents):
        result = "age"
    elif is_name(my_list, sentence, intents):
        result = "name"
    elif is_address(my_list, sentence, intents):
        result = "address"
    else:
        result = "none"
    return result



def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def get_answer(message):
    print(message)
    my_type = get_type(message, intents)
    if my_type != "none":
        res = get_response(my_type, intents)
    elif Is_movie(message, intents):
        res = "movie"
    elif Is_novel(message, intents):
        res = "novel"
    elif Is_weather(message, intents):
        res = "weather"
    elif Is_HE(message, intents):
        res = "HE"
    elif Is_YQ(message, intents):
        res = "YQ"
    elif message == "计算器":
        res = "JS"
    else:
        res = "AI"
    print(res)
    return res


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False) # 生成设备
    parser.add_argument('--temperature', default=1, type=float, required=False) # 生成的temperature
    parser.add_argument('--topk', default=8, type=int, required=False)
    parser.add_argument('--topp', default=0, type=float, required=False)
    parser.add_argument('--log_path', default='data/interact.log', type=str, required=False)
    parser.add_argument('--list_path', default='list/list.txt', type=str, required=False)
    parser.add_argument('--model_path', default='model/epoch30', type=str, required=False)
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False)
    parser.add_argument('--max_len', type=int, default=25) # 每个utterance的最大长度,超过指定长度则进行截断
    parser.add_argument('--max_history_len', type=int, default=3) # max dialogue history
    parser.add_argument('--no_cuda', action='store_true')
    # 重复惩罚参数，若生成的对话重复性较高，可适当提高该参数
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)
    return parser.parse_args()


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


"""
    Learning Note:

    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

        Args:
            logits: logits distribution shape (vocab size)

            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al.
            
            articles: (http://arxiv.org/abs/1904.09751)

            or find it at this way: ./References/The Curious Case of Neural Text Degeneration.pdf

        From: https://github.com/thomwolf?tab=repositories

"""

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    
    # My text is generated one character by another
    
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk() return the last token of the top-k like: (values,indices)
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')  # other elements are all set to -float('Inf')

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
    return logits


def generate(ques):
    args = set_args()
    # store the Info
    logger = create_logger(args)

    # choose the device
    args.cuda = torch.cuda.is_available() and not args.no_cuda and torch.cuda.device_count() >= 1
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    # logging function
    logger.info('using device:{}'.format(device))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tokenizer = BertTokenizerFast(
        vocab_file=args.list_path, 
        sep_token="[SEP]", 
        pad_token="[PAD]", 
        cls_token="[CLS]"
    )

    # load my model
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    
    # initial the record directory and files
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
   
    # save the conversations
    history = []

    try:
        text = ques
        if args.save_samples_path:
            samples_file.write("user:{}\n".format(text))
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        history.append(text_ids)
        input_ids = [tokenizer.cls_token_id]  # input begin with [CLS]

        # get the len num of history dialogs to generate sentence for this time
        for index, history_dialogue in enumerate(history[-args.max_history_len:]):
            input_ids.extend(history_dialogue)
            # sentences are divided by [SEP] (sep_token_id)
            input_ids.append(tokenizer.sep_token_id)
        input_ids = torch.tensor(input_ids).long().to(device)
        
        # clear
        input_ids = input_ids.unsqueeze(0)
        
        # store the output response
        response = []
        # generate the response according to the history_dialogue
        for _ in range(args.max_len):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]

            # give the toke that generated a penalty value
            # if the dailogues generated are highly similar with each other, we can raise this value slightly through the args
            for id in set(response):
                next_token_logits[id] /= args.repetition_penalty
            
            next_token_logits = next_token_logits / args.temperature

            # token can not be [UNK]
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            
            '''
                Learning Note:

                这里用中文做注释，怕说不清楚

                torch.multinomial表示从集合中无放回地抽取num_samples个元素
                说明：权重越高→抽到的几率越高 
                返回元素的下标
                这里做softmax的归一化操作 而后生成运用的是贪心算法
                相当于对于每个中文字符的生成都是根据上下文得出一个可能性最大的值作为输出的结果
                当遇到[SEP]即预示着生成的结束

                top_k_top_p_filtering:

                这个函数的处理思路参照了以下作者thomwolf的仓库内的GPT2的项目源码:
                From: https://github.com/thomwolf?tab=repositories

                论文依据:
                articles: (http://arxiv.org/abs/1904.09751)

            '''
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            if next_token == tokenizer.sep_token_id:  # [SEP] → generate response comes to an end
                break
            response.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
            # print("his_text:{}".format(his_text))
        
        history.append(response)
        text = tokenizer.convert_ids_to_tokens(response)
        if args.save_samples_path:
            samples_file.write("chatbot:{}\n".format("".join(text)))
        return "".join(text)

    except KeyboardInterrupt:
        if args.save_samples_path:
            samples_file.close()
        exit


#从app模块中即从__init__.py中导入创建的app应用
from glob import glob
from app import app

from flask import Flask,jsonify,render_template,request
import json
#建立路由，通过路由可以执行其覆盖的方法，可以多个路由指向同一个方法。

chengshi=[]
shuju=[]
#表格，目前来说只有疫情的爬取用到了这个
@app.route('/getjson2', methods=['GET'])
def getjson2():
    dataAxis = chengshi
    data = shuju
    result = {'dataAxis':dataAxis , 'data':data,}
    return jsonify(result)
#输入的话
data=""
@app.route('/post_text',methods=['POST'])
def post_text():
    global data
    data=request.form.get('mytext')



is_movie = 0
is_weather = 0
is_YQ = 0
is_HE = 0
is_JS = 0

#机器人的话
@app.route('/get_text',methods=['GET'])
def get_text():
    global is_movie
    global is_weather
    global is_YQ
    global is_JS
    res = get_answer(data)
    if is_movie == 1:
        print("is movie")
        test=pick_movie(data)
        is_movie = 0
    elif is_weather == 1:
        test=get_weather(data)
        is_weather = 0
    elif is_YQ == 1:
        print("is_YQ=1")
        test=get_covid(data)
        print(chengshi)
        print(shuju)
        is_YQ = 0
    elif is_JS == 1:
        test=cal(data)
        is_JS = 0
    elif res == "AI":
        test=generate(data)
    elif res == "movie":
        test="有这些类型的电影：剧情 喜剧 动作 爱情 科幻 动画 悬疑 惊悚\n请从中选择一个"
        is_movie = 1
    elif res == "novel":
        test=get_novel()
    elif res == "weather":
        test="输入城市名，我可以告诉你那里今天的天气"
        is_weather = 1
    elif res == "HE":
        print("hotsearch")
        test=get_hotsearch()
        is_HE = 1
        print(test)
    elif res == "YQ":
        test="输入想查询的省份"
        is_YQ = 1
    elif res == "JS":
        print("JSQ")
        test="请输入表达式"
        is_JS = 1
    else:
        test=res
    return test

@app.route('/get_chart',methods=['GET'])
def get_chart():
    if(is_YQ==1):
        chengshi=[]
        shuju=[]
        get_covid("安徽")
        test="1"
    else:
        test="0"
    return test

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')