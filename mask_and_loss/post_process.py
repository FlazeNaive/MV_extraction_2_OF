import json
import numpy as np

with open('data.json', 'r', encoding='utf-8') as f:
    dict = json.load(f)

MV = np.array([.0, .0, .0])
RAFT = np.array([.0, .0, .0])

print(type(dict))
length = len(dict)

for ele in dict:
    print(type(ele))
    for name in ele.keys():
        data = ele[name]
        # print(name, data)
        MV += np.array(data['MV'])
        RAFT += np.array(data['RAFT'])
        # print(MV, RAFT)

print(MV/length, RAFT/length)