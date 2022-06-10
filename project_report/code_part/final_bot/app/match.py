import jieba
import json
import random
from fuzzywuzzy import fuzz
import weather
import pick_movie
import get_novel

intents = json.loads(open('answers.json', encoding='utf-8').read())


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

def is_movie(sentence, intents):
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


def is_novel(sentence, intents):
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


def is_weather(sentence, intents):
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
    my_type = get_type(message, intents)
    if my_type != "none":
        res = get_response(my_type, intents)
    elif is_movie(message, intents):
        res = "movie"
    elif is_novel(message, intents):
        res = "novel"
    elif is_weather(message, intents):
        res = "weather"
    else:
        res = "AI"

    return res
