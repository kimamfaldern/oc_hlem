import pm4py
import pandas as pd
import numpy as np

path = "order-management2.sqlite"
ocel = pm4py.read_ocel2_sqlite(path)
filtered_ocel = pm4py.filter_ocel_object_attribute(ocel, "ocel:type", ["orders", "items", 'packages'], positive=True)
print(filtered_ocel.get_extended_table())
extended_df = filtered_ocel.get_extended_table()
object_types = pm4py.ocel_get_object_types(filtered_ocel)
for obj in object_types:
    str_name = 'ocel:type:' + obj
    extended_df[str_name] = extended_df[str_name].fillna(0)
    extended_df[str_name] = extended_df[str_name].apply(lambda x: list() if x==0 else x)

objects = dict()
for obj in object_types:
    str_name = 'ocel:type:' + str(obj)
    obj_inst = extended_df[str_name].sum()
    for i in obj_inst:
        objects[i]=np.nan

count = 0
need_redo = True
number_exec = 0
while (need_redo): #and number_exec < 10):
    number_exec = 1
    need_redo = False
    for index, row in extended_df.iterrows():
            qcid = []
            all_inst = []
            for obj in object_types:
                str_name = 'ocel:type:' + obj
                instances = row[str_name]
                if instances != []:
                    for i in instances:
                        all_inst.append(i)
                        if not np.isnan(objects[i]):
                            qcid.append(objects[i])
            if qcid == []:
                for i in all_inst:
                    objects[i] = count
                count = count + 1
            else:
                qcid_check = set(qcid)
                if len(qcid_check)>1:
                    new_qcid = min(qcid_check)
                    for i in all_inst:
                        objects[i] = new_qcid
                    need_redo = True
                else:
                    new_qcid = qcid_check.pop()
                    for i in all_inst:
                        objects[i] = new_qcid         

extended_df['new:orders'] = extended_df['ocel:type:orders']
extended_df['new:items'] = extended_df['ocel:type:items']
extended_df['new:packages'] = extended_df['ocel:type:packages']
qcid_list = []
for index, row in extended_df.iterrows():
    for obj in object_types:
        str_name = 'ocel:type:' + obj
        instances = row[str_name]
        for i in instances:
            qcid_list.append(objects[i])
        str2 = 'new:' + obj
        extended_df.at[index, str2] = qcid_list
        qcid_list = []

print(extended_df)  

extended_df['qcid'] = extended_df['new:items']
for index, row in extended_df.iterrows():
    qcid_list = []
    for obj in object_types:
        str2 = 'new:' + obj
        qcid_list = qcid_list + row[str2]
    qcid_check = set(qcid_list)
    if len(qcid_check)!=1:
        print('problem')
    else:
        final_id = qcid_check.pop()
        extended_df.at[index, 'qcid'] = final_id

print(extended_df)
print(extended_df[extended_df['qcid']==0])


#this method only works if the events are ordered according to the control flow
def proc_exec():
    count = 0
    obj_type_count = 0
    instance_count = 0
    for index, row in extended_df.iterrows():
        for obj in object_types:
            obj_type_count = obj_type_count + 1
            str_name = 'ocel:type:' + obj
            instances = row[str_name]
            if instances != []:
                common = np.nan
                for i in instances:
                    instance_count = instance_count + 1
                    qcid = objects[i]
                    if not np.isnan(qcid):
                        if not np.isnan(common):
                            if qcid != common:
                                print('problem')
                                print(qcid)
                                print(common)
                        else:
                            common = qcid
                    else:
                        if obj_type_count==1 and instance_count==1:
                            count + 1
                        objects[i] = count
                instance_count = 0
        obj_type_count = 0