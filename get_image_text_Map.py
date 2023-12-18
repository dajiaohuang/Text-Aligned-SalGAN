import pandas as pd
import json


df = pd.read_excel('saliency/text最终版本.xlsx', sheet_name='部分-实验设置')

image_column = df['image']
category_column = df['描述种类']

my_dic = dict()

text_column = df['text']

for i in range(len(df)):

    image_value = image_column[i]
    category_value = category_column[i]
    
    if category_value == '整体':
        key = str(image_value) + "_0"
        my_dic[key] = ""
        key = str(image_value) + "_1"
        
        
    elif category_value == '非显著':
        key = str(image_value) + "_2"
    else:
        key = str(image_value) + "_3"

    my_dic[key] = text_column[i]


df2 = pd.read_excel('saliency/text最终版本.xlsx', sheet_name='整体')


image_column2 = df2['image']
text_column2 = df2['text']

for i in range(len(df2)):
    image_value2 = image_column2[i]
    
    key2 = str(image_value2)
    my_dic[key2] = text_column2[i]

    key2 = str(image_value2) + "_0"
    my_dic[key2] = ""
    





for key in my_dic:
    print(key, end=" ")
    print(my_dic[key])

print(len(my_dic))



with open('text.json', 'w') as file:
    json.dump(my_dic, file)



