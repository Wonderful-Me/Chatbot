import json
import random

movies = json.loads(open('movies.json', encoding='utf-8').read())


def pick_movie(type):
    list_of_movies = movies["movies"]
    for data in list_of_movies:
        if data["tag"] == type:
            result = random.choice(data['list'])
            return result
    result = "抱歉哈，没有找到这个类型诶"
    return result




def ask_movie():
    print("有这些类型的电影：剧情 喜剧 动作 爱情 科幻 动画 悬疑 惊悚")
    print("请从中选择一个")
    type = input(">>")
    result = pick_movie(type)
    print(result)


if __name__ == '__main__':
    while 1:
        ask_movie()