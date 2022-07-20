import json
import os
from utils import mean_average
dir="6"

with open("./opticalvalue_obj_{}.txt".format(dir),"r")as f:
    lines=f.readlines()
json_list=[]
for line in lines:
    json_str=json.loads(line)
    json_list.append(json_str)
print(type(json_list[0]))
print(json_list)

average_value=mean_average(json_list,block=5)
print(average_value)


for value in average_value:
    normalopticalvalue=float(value/max(average_value))
    with open("./normalopticalvalue_obj_{}.txt".format(dir),"a")as f:
        f.write(str(normalopticalvalue))
        f.write("\n")
