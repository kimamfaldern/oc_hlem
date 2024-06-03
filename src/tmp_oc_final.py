import pm4py
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import networkx as nx
pd.options.mode.chained_assignment = None  # default='warn'


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

    #replace nan values in object lists by []
    for obj in objects:
        str_name = 'ocel:type:' + str(obj)
        df[str_name] = df[str_name].fillna(0)
        df[str_name] = df[str_name].apply(lambda x: list() if x==0 else x)

    new_items = []
    for index, row in df.iterrows():
        if row['ocel:activity'] == 'confirm order' or row['ocel:activity'] == 'pay order':
            new_items.append([])
        else:
            new_items.append(row['ocel:type:items'])
    df['ocel:type:items'] = new_items

    #print(df.head(n=10))
    #print(qcids)
    #print(len(set(qcids)))

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

    #compute successor and predecessor for every event
    following_relations = dict()
    qcid_set = set(qcids)
    print(len(qcid_set))
    for id in qcid_set:
        print(id)
        #following_relations[id] = dict()
        case_df = df.copy()
        case_df = case_df[case_df['ocel:process:execution']==id]
        successor_relations, predecessor_relations = compute_following_relations(case_df, objects, object_handling)
        following_relation = dict()
        following_relation['succ'] = successor_relations
        following_relation['pred'] = predecessor_relations
        following_relations[id] = following_relation
    print(following_relations[0])

    #count all objects per object type activity
    counter_df = pd.DataFrame()
    counter_df = df.copy()
    for obj in objects:
        str1 = 'dict:' + str(obj)
        str2 = 'ocel:type:' + str(obj)
        counter_df[str1] = counter_df[str2].apply(lambda x: Counter(x))

    #define time bounds
    min_time = min(df['ocel:timestamp'])
    max_time = max(df['ocel:timestamp'])
    time_window = max_time-min_time
    number_segments = 428 #roughly days for order-management dataset
    time_segment = time_window/number_segments
    time_bounds = []
    print(min_time)
    print(max_time)

    for i in range(number_segments):
        time_bounds.append(min_time + (i+1)*time_segment)
    #print(time_bounds)

    #partition df according to time bounds
    counter_df.sort_values("ocel:timestamp", inplace=True)    
    tmp_df_dict = {}
    for j in range(number_segments-1):
        print('execution: ' + str(j))
        str_name = 'tmp_df' + str (j)
        tmp_df_dict[str_name] = counter_df[counter_df['ocel:timestamp']>time_bounds[j]]
        tmp_df_dict[str_name] = counter_df[counter_df['ocel:timestamp']<=time_bounds[j+1]]
        mine_hle(tmp_df_dict[str_name], min_time, time_segment, j, objects, object_handling, following_relations)
        j = j+1

def mine_hle(df, min_time, time_segment, curr_seg_count, objects, object_handling, following_relations):
    #count of objects within object type for whole df (time segment)
    count_dict = {}    
    for obj in objects:
        count_dict[str(obj)] = Counter({})
    for index, row in df.iterrows():
        for obj in objects:
            str_name = 'dict:' +str(obj)
            count_dict[str(obj)].update(row[str_name])
    #find objects with highest demand
    #initialize max_ with highest count 
    #initialieze list to collect all objects that have highest count
    max_dict = {}
    max_list_dict = {}
    for obj in objects:
        if len(count_dict[str(obj)])>0: #if added because of error in max() if empty sequence
            max_dict[str(obj)] = count_dict[str(obj)][max(count_dict[str(obj)], key=count_dict[str(obj)].get)]
            max_list_dict[str(obj)] = []
    #collect all items which have the highest count
    for obj in objects:
        for key, value in count_dict[str(obj)].items():
            if value==max_dict[str(obj)]:
                max_list_dict[str(obj)].append(key)
    
    #initialize lists for high level log
    event = []
    event_type = []
    frequency = []
    start_time = []
    end_time = []
    time_frame = []
    qcid = []
    related_object = []

    #use high demand events for objects types where the instance matters (eg ressource) and high load events for object types where the instance doesnt matter (eg order number)
    #add high level events related to high demand of one object to log
    for obj in objects:
        if not object_handling[obj]:
            counter_of_obj = count_dict[obj]
            obj_inst = list(counter_of_obj)
            for instance in obj_inst:
                event.append('high demand of: ' + str(instance))
                event_type.append('hd')
                frequency.append(counter_of_obj[instance])
                start_time.append(min_time + (curr_seg_count-1)*time_segment)
                end_time.append(min_time + curr_seg_count*time_segment)
                time_frame.append(curr_seg_count)
                qcid.append(np.nan)
                related_object.append(instance)
    #add high level events related to load of one object type to log (only occurence of object type counts)
    for obj in objects:
        if object_handling[obj]:
            counter_of_obj = count_dict[obj]
            type_count = counter_of_obj.total()
            event.append('high load of object type ' + str(obj))
            event_type.append('hl')
            frequency.append(type_count)
            start_time.append(min_time + (curr_seg_count-1)*time_segment)
            end_time.append(min_time + curr_seg_count*time_segment)
            time_frame.append(curr_seg_count)
            qcid.append(np.nan)
            related_object.append(obj)
    
    #count different object types
    sum_column = []
    for index, row in df.iterrows():
        sum = 0
        for obj in objects:
            str_name = 'dict:' + str(obj)
            if row[str_name]!=[]:
                sum = sum +1
        sum_column.append(sum)
    df['num_object_types'] = sum_column
    
    max_object_types = max(sum_column)
    max_df = df[df.num_object_types==max_object_types]

    #add high level event of high load of object types to log
    event.append('high load (over all object types)')
    event_type.append('hol')
    frequency.append(max_df.shape[0])
    start_time.append(min_time + (curr_seg_count-1)*time_segment)
    end_time.append(min_time + curr_seg_count*time_segment)
    time_frame.append(curr_seg_count)
    qcid.append(np.nan)
    related_object.append(np.nan)
 
    #explore waiting times and split/joins
    cases = df['ocel:process:execution'].unique()
    for case in cases:
        relations = following_relations[case]
        predecessor = relations['pred']
        successor = relations['succ']
        case_df = df[df['ocel:process:execution']==case]
        #if events with same objects happen at same time, merge them to one event for further computations 
        #case_df = preprocess_case_df(case_df, objects)
        for index, row in case_df.iterrows():
            current_event = row['ocel:eid']
            current_pred = predecessor[current_event]
            current_succ = successor[current_event]
            pred_df = pd.DataFrame()
            for eid in current_pred:
                tmp_pred = case_df.copy()
                tmp_pred = tmp_pred[tmp_pred['ocel:eid']==eid]
                pred_df = pd.concat([pred_df,tmp_pred])
            succ_df = pd.DataFrame()
            for eid in current_succ:
                tmp_succ = case_df.copy()
                tmp_succ = tmp_succ[tmp_succ['ocel:eid']==eid]
                succ_df = pd.concat([succ_df,tmp_succ])
            if(current_succ != []):
                succ_act = set(succ_df['ocel:activity'])
                #check if object leavs the process
                if len(succ_act)==1:
                    curr_obj = row['all objects'].copy()
                    succ_obj = []
                    for si, srow in succ_df.iterrows():
                        for sinst in srow['all objects']:
                            succ_obj.append(sinst)
                    for succ_o in succ_obj:
                        if succ_o in curr_obj:
                            curr_obj.remove(succ_o)
                    if curr_obj!=[]:
                        for inst in curr_obj:
                            if inst in objects:
                                #add high level event of object type leave to log
                                event.append('object type ' + str(inst) + ' leaves the process in activity ' + str(row['ocel:activity']))
                                event_type.append('tl')
                                frequency.append(1)
                                #TODO: check times
                                start_time.append(min_time + (curr_seg_count-1)*time_segment)
                                end_time.append(min_time + curr_seg_count*time_segment)
                                time_frame.append(curr_seg_count)
                                qcid.append(case)
                                related_object.append(inst)
                            else:
                                #add high level event of object leave to log
                                event.append('object ' + str(inst) + ' leaves the process in activity ' + str(row['ocel:activity']))
                                event_type.append('ol')
                                frequency.append(1)
                                #TODO: check times
                                start_time.append(min_time + (curr_seg_count-1)*time_segment)
                                end_time.append(min_time + curr_seg_count*time_segment)
                                time_frame.append(curr_seg_count)
                                qcid.append(case)
                                related_object.append(inst)
                elif len(succ_act)>1:
                    event.append('object (type) split in activity ' + str(row['ocel:activity']) + ' with object (types) ' + str(row['all objects']))
                    event_type.append('s')
                    frequency.append(len(row['all objects']))
                    #TODO: check times
                    start_time.append(min_time + (curr_seg_count-1)*time_segment)
                    end_time.append(min_time + curr_seg_count*time_segment) 
                    time_frame.append(curr_seg_count)
                    qcid.append(case)
                    related_object.append(row['all objects'])
            if current_pred!=[]:
                if len(current_pred)>1:
                    #get execution time and predecessor time and index
                    exec_time = row['ocel:timestamp']
                    early_time = min(pred_df['ocel:timestamp'])
                    early_index = pred_df[pred_df['ocel:timestamp']==early_time].index
                    late_time = max(pred_df['ocel:timestamp'])
                    late_index = pred_df[pred_df['ocel:timestamp']==early_time].index
                    #lagging and delay time are based on opera measures but need to be calculated manually because of different log format
                    lagging_time = late_time - early_time
                    delay_time = exec_time - late_time
                    #add high level event of high lagging time per activity to log
                    for i in late_index:
                        #TODO: anpassen
                        lagging_obj = pred_df.loc[i, 'all objects'].copy()
                        all_curr_obj = row['all objects'].copy()
                        lagging_obj_result = []
                        for il in lagging_obj:
                            if il in all_curr_obj:
                                lagging_obj_result.append(il)
                        lagging_o = lagging_obj_result.pop()
                        if lagging_o in objects:
                            #object type
                            event.append('high lagging time of object type' + lagging_o + ' for activity ' + str(row['ocel:activity']))
                            event_type.append('lt')
                            frequency.append(lagging_time)
                            start_time.append(min_time + (curr_seg_count-1)*time_segment)
                            end_time.append(min_time + curr_seg_count*time_segment)
                            time_frame.append(curr_seg_count)
                            qcid.append(case)
                            related_object.append(lagging_o)
                        else:
                            #object instance
                            event.append('high lagging time of object ' + str(lagging_o) + ' for activity ' + str(row['ocel:activity']))
                            event_type.append('lt')
                            frequency.append(lagging_time)
                            start_time.append(min_time + (curr_seg_count-1)*time_segment)
                            end_time.append(min_time + curr_seg_count*time_segment)
                            time_frame.append(curr_seg_count)
                            qcid.append(case)
                            related_object.append(lagging_o)
                #discover joins of object types
                pred_act = set(pred_df['ocel:activity'])
                #test if join happens
                #check if object type joins
                if len(pred_act)==1:
                    curr_obj = row['all objects'].copy()
                    succ_obj = []
                    for si, srow in succ_df.iterrows():
                        for sinst in srow['all objects']:
                            succ_obj.append(sinst)
                    for succ_o in succ_obj:
                        if succ_o in curr_obj:
                            curr_obj.remove(succ_o)
                    if curr_obj!=[]:
                        for inst in curr_obj:
                            if inst in objects:
                                #add high level event of object type join to log
                                event.append('object type ' + str(inst) + ' joins the process in activity ' + str(row['ocel:activity']))
                                event_type.append('tj')
                                frequency.append(1)
                                #TODO: check times
                                start_time.append(min_time + (curr_seg_count-1)*time_segment)
                                end_time.append(min_time + curr_seg_count*time_segment)
                                time_frame.append(curr_seg_count)
                                qcid.append(case)
                                related_object.append(inst)
                            else:
                                #add high level event of object join to log
                                event.append('object ' + str(inst) + ' joins the process in activity ' + str(row['ocel:activity']))
                                event_type.append('oj')
                                frequency.append(1)
                                #TODO: check times
                                start_time.append(min_time + (curr_seg_count-1)*time_segment)
                                end_time.append(min_time + curr_seg_count*time_segment)
                                time_frame.append(curr_seg_count)
                                qcid.append(case)
                                related_object.append(inst)
                #check if multiple activities are joint
                elif len(pred_act)>1:
                    event.append('object (type) join in activity ' + str(row['ocel:activity']) + ' with objects ' + str(row['all objects']))
                    event_type.append('j')
                    frequency.append(len(row['all objects']))
                    #TODO: check times
                    start_time.append(min_time + (curr_seg_count-1)*time_segment)
                    end_time.append(min_time + curr_seg_count*time_segment)  
                    time_frame.append(curr_seg_count) 
                    qcid.append(case)
                    related_object.append(row['all objects'])                

    #convert to dataframe
    d = {'event': event, 'event type': event_type, 'frequency': frequency, 'start time': start_time, 'end time': end_time, 'time segment': time_frame, 'quasi case id': qcid, 'related object': related_object}
    hl_log = pd.DataFrame(data=d)
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    save_name = parent_path + '/hl_logs_order-managemant/hl_log' + str(curr_seg_count) + '.csv'
    hl_log.to_csv(save_name)
    print(hl_log)

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

def own_del(x):
    del x[::2]
    return x

def own_set_del(x):
    if '[' in x:
        x.remove('[')
    if ']' in x:
        x.remove(']')
    if ']]' in x:
        x.remove(']]')
    if '[[' in x:
        x.remove('[[')
    if '], ' in x:
        x.remove('], ')
    if '], [' in x:
        x.remove('], [')
    if ', [' in x:
        x.remove(', [')
    if ', ' in x:
        x.remove(', ')
    return x

def list_intersect(x, s, row):
    return_list = []
    for i in row[s]:
        if i in x:
            return_list.append(i)
    return return_list

def pattern_detection_dict2():
    df = pd.DataFrame()
    #objects = ['goods receipt', 'invoice receipt', 'material', 'payment', 'purchase_order', 'purchase_requisition', 'quotation']
    #object_handling = {'goods receipt': True, 'invoice receipt': True, 'material': True, 'payment': True, 'purchase_order':True, 'purchase_requisition': True, 'quotation': True}
    objects = ['products', 'orders', 'items', 'employees', 'customers', 'packages']
    object_handling = dict()
    object_handling['orders'] = True
    object_handling['items'] = True
    object_handling['packages'] = True
    object_handling['products'] = False
    object_handling['employees'] = False
    object_handling['customers'] = False
    
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    for i in range(427):
        str_name = parent_path + '/hl_logs_order-management/hl_log' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)

    df['id'] = df.index
    print('number of events:')
    print(len(df))

    #filter high overall load events by just keeping the above median freuquent ones
    df_hol = df[df['event type']=='hol']
    print(len(df_hol))
    df_hol.drop_duplicates(subset=['event', 'time segment'], inplace=True, keep='first')
    print('all high over all load events:')
    print(len(df_hol))
    df_hol = df_hol.astype({'frequency':'int'})
    df_hol = df_hol.sort_values(by=['frequency'])
    half = int(len(df_hol)/2)
    df_hol.drop(df_hol.head(half).index,inplace=True)
    print(df_hol)
    print('high over all load events after filter:')
    print(len(df_hol))

    print('start own_del')
    print(df.head(n=10))
    df.dropna(subset=['related object'], inplace=True)
    df['related object'] = df['related object'].apply(lambda x: x.split('\''))
    df['related object'] = df['related object'].apply(lambda x: set(x))
    df['related object'] = df['related object'].apply(lambda x: own_set_del(x))
    df['related object'] = df['related object'].apply(lambda x: list(x))
    print('finished own_de')
    print(df.head(n=10))
    print(len(df))

    #filter high load events by just keeping the highest loads per object type (above mittelwert)
    df_hl = df.copy()
    df_hl = df_hl[df_hl['event type']=='hl']
    print(len(df_hl))
    df_hl.drop_duplicates(subset=['event', 'time segment'], inplace=True, keep='first')
    print('all high load events:')
    print(len(df_hl))
    print(df_hl)
    df_hl = df_hl.astype({'frequency':'int'})
    filtered_df_hl = pd.DataFrame()
    for obj in objects:
        df_hl_obj = df_hl.copy()
        df_hl_obj['is related object'] = df_hl_obj['related object'].apply(lambda x: obj in x)
        df_hl_obj = df_hl_obj[df_hl_obj['is related object']==True]
        df_hl_obj.drop(columns='is related object')
        #print(obj)
        #print(len(df_hl_obj))
        if len(df_hl_obj['frequency'])>0:
            max_obj = int(max(df_hl_obj['frequency']))
        df_hl_obj['frequency'] = df_hl_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))
        filtered = df_hl_obj[df_hl_obj['frequency']==True]
        #filtered = df_hl_obj[df_hl_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))]
        filtered_df_hl = pd.concat([filtered_df_hl, filtered])
    #filtered_df_hl = df_hl.copy()
    print('high load events after filtering:')
    print(len(filtered_df_hl))
    #print(filtered_df_hl)
        
    #filter high demand events by just keeping the highest demands per object
    #TODO: adapt to PER OBJECT
    df_hd = df.copy()
    df_hd = df_hd[df_hd['event type']=='hd']
    print(len(df_hd))
    df_hd.drop_duplicates(subset=['event', 'time segment'], inplace=True, keep='first')
    print('all high demand events:')
    print(len(df_hd))
    df_hd = df_hd.astype({'frequency':'int'})
    filtered_df_hd = df_hd.sort_values('frequency', ascending=False)
    num_hd = len(df_hd)
    filtered_df_hd = filtered_df_hd.head(n=int(num_hd/2))
    filtered_df_hd = df_hd.copy()
    print('high demand events after filter:')
    print(len(filtered_df_hd))
    #print(filtered_df_hd)

    #filter for split of objects in events
    df_s = df.copy()
    df_s = df_s[df_s['event type']=='s']
    print('split events:')
    print(len(df_s))
    #print(df_s)

    #filter join of object type events
    df_m = df.copy()
    df_m = df_m[df_m['event type']=='j']
    df_m['event type'] = 'm'
    print('join in event events:')
    print(len(df_m))
    print(df_m['related object'])

    #filter for object leaves
    df_l = df.copy()
    df_l1 = df_l[df_l['event type']=='tl']
    df_l2 = df_l[df_l['event type']=='ol']
    df_l = pd.concat([df_l1, df_l2])
    print('leave events:')
    print(len(df_l))
    df_l['event type'] = 'l'

    #filter for object joins
    df_j = df.copy()
    df_j1 = df_j[df_j['event type']=='tj']
    df_j2 = df_j[df_j['event type']=='oj']
    df_j = pd.concat([df_j1, df_j2])
    print('join process events:')
    print(len(df_j))
    df_j['event type'] = 'j'

    #filter lagging time events by keeping those below 0.7*max lagging time
    df_llt = df.copy()
    df_llt = df_llt[df_llt['event type']=='lt']
    print('all lagging time events:')
    print(len(df_llt))
    df_llt.sort_values('frequency', ascending=True, inplace=True)
    number_lt = len(df_llt)
    filtered_llt = df_llt.head(n=int(number_lt/10))
    filtered_llt['event type'] = 'llt'
    print('low lagging time events after filter:')
    print(len(filtered_llt))

    #filter lagging time events by keeping those above 0.7*max lagging time
    df_lt = df.copy()
    df_lt = df_lt[df_lt['event type']=='lt']
    df_lt.sort_values('frequency', ascending=False, inplace=True)
    number_lt = len(df_lt)
    filtered_lt = df_lt.head(n=int(number_lt/10))
    filtered_lt['event type'] = 'hlt'
    print('high lagging time events after filter:')
    print(len(filtered_lt))

    new_df = pd.concat([filtered_df_hl, filtered_df_hd, df_s, df_m, df_l, df_j, filtered_llt, filtered_lt], ignore_index=True)#len=16230, ohne s,m,l,j: len=1231
    #print(len(new_df_obj))

    #look for patterns
    #compute all object instances/types that cause at least one hle
    all_objects = set()
    for index, row in new_df.iterrows():
        tmp_obj = row['related object']
        for obj in tmp_obj: 
            all_objects.add(obj)
    print(all_objects)
    print(len(all_objects))
    #create df for the result standard event log with classical case ids
    result_dict = dict()
    cascade_dict = dict()
    all_ids = new_df['id'].unique()
    for id in all_ids:
        result_dict.setdefault(id, [])
        cascade_dict.setdefault(id, [])
    #look for patterns
    new_df2 = new_df.copy()
    print(len(new_df))
    count = 0
    for index,row in new_df.iterrows():
        count = count + 1
        print(count)
        time1 = row['time segment']
        tw_df = new_df2.copy()
        tw_df = tw_df[tw_df['time segment'] == time1+1]
        for i,r in tw_df.iterrows():
            objects1 = row['related object']
            objects2 = r['related object']
            shared_objects = [value for value in objects1 if value in objects2]
            if not(shared_objects == []):
                #print('shared objects exist.')
                #print(shared_objects)
                type1 = row['event type']
                type2 = r['event type']
                #check for hd->hlt, hl->hlt, s->hlt, m->hlt
                first_type = ['hd', 'hl', 's', 'm']
                if (type1 in first_type and type2 == 'hlt'):
                    first_id = row['id']
                    second_id = r['id']
                    succ_list = result_dict[first_id]
                    succ_list.append(second_id)
                    result_dict[first_id] = succ_list
            case1 = row['quasi case id']
            case2 = r['quasi case id']
            if case1 == case1:
                #check for j->hlt, m->hlt
                type1 = row['event type']
                type2 = r['event type']
                if (type1 == 'j' and type2 =='hlt'):
                    first_id = row['id']
                    second_id = r['id']
                    succ_list = result_dict[first_id]
                    succ_list.append(second_id)
                    result_dict[first_id] = succ_list
                #look for chains of 'j', 's', 'l', 'm'
                first_type = ['j', 's', 'l', 'm']
                if (type1 in first_type and type1 == type2):
                    first_id = row['id']
                    second_id = r['id']
                    succ_list = cascade_dict[first_id]
                    succ_list.append(second_id)
                    cascade_dict[first_id] = succ_list
                #check for l->llt, hlt->l->llt
                #check for s->llt, hlt->s->llt
                first_type = ['l', 's']
                if (type1 in first_type and type2 == 'llt'):
                    first_id = row['id']
                    second_id = r['id']
                    succ_list = result_dict[first_id]
                    succ_list.append(second_id)
                    result_dict[first_id] = succ_list

    #result_df['case id'] = case_ids
    result_size = 0
    cascade_size = 0
    node_set = set()
    node_dict = result_dict.copy()
    for id in all_ids:
        if not(result_dict[id] == []):
            node_set.add(id)
            for x in result_dict[id]:
                node_set.add(x)
            print(id)
            print(result_dict[id])
            result_size = result_size +1
        else:
            del node_dict[id]
    print('result size: ' + str(result_size))
    for id in all_ids:
        if not(cascade_dict[id] == []):
            #print(result_dict[id])
            cascade_size = cascade_size +1
    print('cascade size: ' + str(cascade_size))
    #print(result_dict)
    #print(cascade_dict)

    G1 = nx.DiGraph(result_dict)
    #nx.draw(G1)
    #plt.savefig('G1')
    G2 = nx.DiGraph(node_dict)
    #nx.draw(G2)
    #plt.savefig('G2')
    G3  = nx.DiGraph(cascade_dict)
    #nx.draw(G3)
    #plt.savefig('G3')
    nx.write_edgelist(G2, './graphs/G2.edgelist')
    nx.write_edgelist(G1, './graphs/G1.edgelist')
    nx.write_edgelist(G3, './graphs/G3.edgelist')
    #G = nx.DiGraph()



main()
#pattern_detection_dict2()
#case_construction()