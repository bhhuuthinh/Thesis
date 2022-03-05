import json
import pandas as pd

list = {}

videos = ['001','002','003']
list_names = [['a','b','c'],
        ['a','c'],
        ['c','a']]

for idx_video, names in enumerate(list_names):
    for idx_name, name in enumerate(names):
        if name not in list:
            list[name]=[]
        list[name].append(videos[idx_video])
