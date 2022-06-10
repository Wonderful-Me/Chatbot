import json
import random
novels = json.loads(open('novel.json', encoding='utf-8').read())

def get_novel():
    novel_list = novels["novels"]
    result = random.choice(novel_list)
    return result

