import pm4py
import os
import numpy as np

curr_path = os.getcwd()
parent_path = os.path.dirname(curr_path)
root_path = os.path.dirname(parent_path)
dataset = './input_data/order-management2'
path = root_path + dataset + '.sqlite'
print(path)
log = pm4py.read_ocel2_sqlite(path)
df = log.get_extended_table() #extended_df
objects = pm4py.ocel_get_object_types(log)
object_handling= dict()
if dataset == './input_data/order-management2':
    object_handling['orders'] = True
    object_handling['items'] = True
    object_handling['packages'] = True
    object_handling['products'] = False
    object_handling['employees'] = False
    object_handling['customers'] = False
flow_obj = []
for obj in objects:
    if object_handling[obj]:
        flow_obj.append(obj)

filtered_ocel = pm4py.filter_ocel_object_attribute(log, "ocel:type", flow_obj, positive=True)
extended_df = filtered_ocel.get_extended_table()
for obj in flow_obj:
    str_name = 'ocel:type:' + obj
    extended_df[str_name] = extended_df[str_name].fillna(0)
    extended_df[str_name] = extended_df[str_name].apply(lambda x: list() if x==0 else x)

all_objects = dict()
for obj in flow_obj:
    str_name = 'ocel:type:' + str(obj)
    obj_inst = extended_df[str_name].sum()
    for i in obj_inst:
        all_objects[i]=np.nan

count = 0
need_redo = True
number_exec = 0
while (need_redo): #and number_exec < 10):
    number_exec = number_exec + 1
    need_redo = False
    for index, row in extended_df.iterrows():
        qcid = []
        all_inst = []
        for obj in flow_obj:
            str_name = 'ocel:type:' + obj
            instances = row[str_name]
            if instances != []:
                for i in instances:
                    all_inst.append(i)
                    if not np.isnan(all_objects[i]):
                        qcid.append(all_objects[i])
        if qcid == []:
            for i in all_inst:
                all_objects[i] = count
            count = count + 1
        else:
            qcid_check = set(qcid)
            if len(qcid_check)>1:
                new_qcid = min(qcid_check)
                for i in all_inst:
                    all_objects[i] = new_qcid
                need_redo = True
            else:
                new_qcid = qcid_check.pop()
                for i in all_inst:
                    all_objects[i] = new_qcid  

for obj in flow_obj:
    str_name = 'new:' + obj
    str_name2 = 'ocel:type:' + obj
    extended_df[str_name] = extended_df[str_name2]

qcid_list = []
for index, row in extended_df.iterrows():
    for obj in flow_obj:
        str_name = 'ocel:type:' + obj
        instances = row[str_name]
        for i in instances:
            qcid_list.append(all_objects[i])
        str2 = 'new:' + obj
        extended_df.at[index, str2] = qcid_list
        qcid_list = []

    #extended_df['qcid'] = extended_df['ocel:eid']
qcids = []
eid_dict = dict()
for index, row in extended_df.iterrows():
    qcid_list = []
    for obj in flow_obj:
        str2 = 'new:' + obj
        qcid_list = qcid_list + row[str2]
    qcid_check = set(qcid_list)
    if len(qcid_check)!=1:
        print('problem')
    else:
        final_id = qcid_check.pop()
        #extended_df.at[index, 'qcid'] = final_id
        qcids.append(final_id)
        eid = row['ocel:eid']
        eid_dict[eid] = final_id
extended_df['qcid'] = qcids

extended_df_test = extended_df[extended_df['qcid']==0]
save_name = 'df_test.csv'
extended_df_test.to_csv(save_name)
