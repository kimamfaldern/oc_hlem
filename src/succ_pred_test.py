import pm4py
import os    
import numpy as np
import pandas as pd

def main():
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    dataset = 'order-management2'
    path = parent_path + './input_data/' + dataset + '.sqlite'
    print(path)
    log = pm4py.read_ocel2_sqlite(path)
    df = log.get_extended_table() #extended_df
    objects = pm4py.ocel_get_object_types(log)
    print(objects)
    #set handling of object type: True=only type matters, not single instance, False=instance matters
    object_handling = {}
    if dataset == './procure-to-pay':
        for obj in objects:
            object_handling[obj] = True
    if dataset == 'order-management2':
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
    print('all activity types:')
    print(extended_df['ocel:activity'].unique())

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

    #print(extended_df)
    qcids = []
    for index, row in df.iterrows():
        e = row['ocel:eid']
        qcids.append(eid_dict[e])
    #print(qcids)
    df['ocel:process:execution'] = qcids
    print_df = df.copy()
    print_df = print_df[print_df['ocel:process:execution']==0]
    print_df1 = print_df.copy()
    print_df1 = print_df1[print_df1['ocel:activity'] == 'pick item']
    print_df2 = print_df.copy()
    print_df2 = print_df2[print_df2['ocel:activity'] == 'confirm order']
    print_df3 = print_df.copy()
    print_df3 = print_df3[print_df3['ocel:activity'] == 'place order']
    print_df12 = pd.concat([print_df1, print_df2, print_df3])
    print_df12.sort_values('ocel:timestamp', inplace=True)
    print(print_df[['ocel:eid', 'ocel:type:items', 'ocel:type:orders', 'ocel:type:packages']])

    #replace nan values in object lists by []
    for obj in objects:
        str_name = 'ocel:type:' + str(obj)
        df[str_name] = df[str_name].fillna(0)
        df[str_name] = df[str_name].apply(lambda x: list() if x==0 else x)

    #print(df.head(n=10))
    #print(qcids)
    #print(len(set(qcids)))
        
    new_items = []
    for index, row in df.iterrows():
        if row['ocel:activity'] == 'confirm order' or row['ocel:activity'] == 'pay order' or row['ocel:activity'] == 'payment reminder':
            new_items.append([])
        else:
            new_items.append(row['ocel:type:items'])
    df['ocel:type:items'] = new_items

    all_obj_col = []
    for index,row in df.iterrows():
        all_obj_list = []
        for obj in objects:
            str_name = 'ocel:type:' + str(obj)
            if object_handling[obj]:
                for inst in row[str_name]:
                    all_obj_list.append(obj)
            else:
                for inst in row[str_name]:
                    all_obj_list.append(inst)
        all_obj_col.append(all_obj_list)
    df['all objects'] = all_obj_col


    #print(df[['ocel:type:orders', 'ocel:type:items', 'ocel:type:packages', 'ocel:type:products', 'ocel:type:employees', 'ocel:type:customers', 'all objects']].head(n=10))
    print(df[['ocel:activity', 'all objects']].head(n=10))
    pick_i_df = df.copy()
    pick_i_df = pick_i_df[pick_i_df['ocel:activity']=='pick item']
    print('pick i df:')
    print(pick_i_df[['ocel:type:orders', 'ocel:type:items', 'ocel:type:packages', 'ocel:type:products', 'ocel:type:employees', 'ocel:type:customers']].head(n=10))

    #compute successor and predecessor for every event
    following_relations = dict()
    qcid_set = set(qcids)
    for id in qcid_set:
        #print(id)
        if id==0:
            #following_relations[id] = dict()
            case_df = df.copy()
            case_df = case_df[case_df['ocel:process:execution']==id]
            #print(case_df)
            successor_relations, predecessor_relations = compute_following_relations(case_df, objects, object_handling)
            following_relation = dict()
            following_relation['succ'] = successor_relations
            following_relation['pred'] = predecessor_relations
            following_relations[id] = following_relation
    succ_and_pre = following_relations[0]
    pre = succ_and_pre['pred']
    suc = succ_and_pre['succ']
    print(pre['pick_i-880006'])
    print(suc['pick_i-880006'])
    print(pre['confirm_o-990002'])
    print(suc['confirm_o-990002'])
    print(pre['create_p-660001'])
    print(suc['create_p-660001'])
    #print(following_relations[0])

def compute_following_relations(df, objects, object_handling):
    succ_relations = dict()
    pre_relations = dict()
    all_ids = df['ocel:eid'].unique()
    for id in all_ids:
        succ_relations.setdefault(id, [])
        pre_relations.setdefault(id, [])
    save_df = df.copy()
    list_of_all_objects = []
    for index, row in save_df.iterrows():
        all_objects = []
        for obj in objects:
            if object_handling[obj]:
                str_name = 'ocel:type:' + str(obj)
                curr_objects = row[str_name]
                for i in curr_objects:
                    all_objects.append(i)
        list_of_all_objects.append(all_objects)
    save_df['all objects tmp'] = list_of_all_objects
    for index, row in save_df.iterrows():
        save_df.drop(index, inplace=True)
        for i, r in save_df.iterrows():
            curr_objects = row['all objects tmp']
            if curr_objects != list():
                succ_obj = r['all objects tmp']
                intersecting_obj = [value for value in curr_objects if value in succ_obj]
                if not(intersecting_obj == []):
                    first_eid = row['ocel:eid']
                    second_eid = r['ocel:eid']
                    new_succ = succ_relations[first_eid].copy()
                    new_succ.append(second_eid)
                    succ_relations[first_eid] = new_succ.copy()
                    new_pre = pre_relations[second_eid].copy() 
                    new_pre.append(first_eid)
                    pre_relations[second_eid] = new_pre.copy()
                    for o in intersecting_obj:
                        curr_objects.remove(o)
                    row['all objects tmp'] = curr_objects
    return succ_relations, pre_relations

main()
