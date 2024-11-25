import numpy as np
import os
import pandas as pd

def list_del():
    test_list = ['[i1 ', 'i2', ' i3] ']
    print (test_list)
    new_list = []
    for x in test_list:
        y = x.strip()
        y = y.replace('[', '')
        new_list.append(y)
    print(new_list)

    test_string = ' [i1 '
    print(x)
    y = x.strip()
    print(y)

def list_containment_test():
    data = {'col1': [[1,2,3], [1], [2,3,4,5,6]], 'col2':[1,2,3]}
    df = pd.DataFrame(data=data)
    print(df)
    for index,row in df.iterrows():
        if row['col1']==[1]:
            df.loc[index, 'col2'] = 99
    print(df)
    new_df = df.copy()
    new_df['col1'] = new_df['col1'].apply(lambda x: list_contain(x,1))
    print(new_df)
    new_new_df = new_df[new_df['col1'] != 'nan']
    print(new_new_df)

def list_contain(list, val):
    if val in list:
        return list
    else: 
        return 'nan'
    
def list_len_test():
    my_list = [1,2,3,4]
    print(len(my_list))
    print(range(4))
    for i in range(len(my_list)):
        print(i)
        print(my_list[i])

def dict_test():
    my_dict = dict()
    my_dict[1] = [1,2,3]
    my_dict[2] = [4,5,6]
    print(my_dict)
    keys = list(my_dict.keys())
    for k in keys:
        print(k)
        print(my_dict[k])

list_containment_test()