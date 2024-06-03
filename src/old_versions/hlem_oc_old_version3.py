import pm4py
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import networkx as nx
import pickle
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
    if dataset == './input_data/procure-to-pay':
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

    #print(df)
    #start_act = {}
    #end_act = {}
    #for obj in objects:
    #    obj_log = pm4py.ocel_flattening(log, obj)
    #    log_start = pm4py.get_start_activities(obj_log)
    #    start_act.update(log_start)
    #    log_end = pm4py.get_end_activities(obj_log)
    #    end_act.update(log_end)
    #print(list(start_act.keys()))
    #print(list(end_act.keys()))

    #count all objects per object type activity
    counter_df = pd.DataFrame()
    counter_df = df.copy()
    for obj in objects:
        str1 = 'dict:' + str(obj)
        str2 = 'ocel:type:' + str(obj)
        #replace nan by []
        counter_df[str2] = counter_df[str2].fillna(0)
        counter_df[str2] = counter_df[str2].apply(lambda x: list() if x==0 else x)
        #print(counter_df[str2])
        counter_df[str1] = counter_df[str2].apply(lambda x: Counter(x))

    #look for object patterns in activities
    #not for every time segment but over whole log
    object_type_activities = pm4py.ocel_object_type_activities(log)
    all_activities = []
    for obj in objects:
        for inst in object_type_activities[obj]:
            if inst not in all_activities:
                all_activities.append(inst)
    #all_activities = pm4py.get_event_attribute_values(log, "ocel:activity")
    #for key in all_activities:
    #    #define df only with one activity
    #    act_df = counter_df.copy(deep=True)
    #    #alibi na check
    #    act_df['ocel:activity'] = act_df['ocel:activity'].fillna(0)
    #    act_df['ocel:activity'] = act_df['ocel:activity'].apply(lambda x: x if x==key else np.nan)
    #    act_df.dropna(subset=['ocel:activity'], inplace=True)
    #    #one_act_log = pm4py.filter_event_attribute_values(log, "concept:name", [key], level="event", retain=True)
    #    search_patterns(act_df, objects, key)

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
        mine_hle(tmp_df_dict[str_name], min_time, time_segment, j, objects, object_handling)
    #tmp_df = pd.DataFrame()
    #for index, row in counter_df.iterrows():    #probably better to use apply or something
    #    if row["ocel:timestamp"] <= time_bounds[j]:
    #        tmp_df = pd.concat([tmp_df, row.to_frame().T])
    #    else:
    #        j=j+1
    #        print('objects in call:')
    #        print(objects)
    #        mine_hle(tmp_df, min_time, time_segment, j, objects, object_handling)
    #        tmp_df = pd.DataFrame()

def mine_hle(df, min_time, time_segment, curr_seg_count, objects, object_handling):
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
    #print('cases:')
    #print(cases)
    #print(len(cases))
    for case in cases:
        case_df = df[df['ocel:process:execution']==case]
        #if events with same objects happen at same time, merge them to one event for further computations 
        case_df = preprocess_case_df(case_df, objects)
        #create copy to have one for successor and one for prdecessor calculation because this calculation manipulate the df
        case_df_copy = case_df.copy()
        #sort by timestamp
        for index, row in case_df_copy.iterrows():
            successor = get_successor(row,index, case_df_copy, objects)
            #HINT: complete df is spitted according to time bounds. This leads to different start/end points of cases. Therefor some splits or joins are not/wrong found.
            if(str(successor)!='NaN'):
                succ_act = set(successor['ocel:activity'])
                #check if object leavs the process
                if len(succ_act)==1:
                    #TODO: theoretical change set to list
                    curr_obj = set()
                    curr_obj_type = {}
                    for obj in objects:
                        str_name = 'ocel:type:' + str(obj)
                        if row[str_name]!=[]:
                            for inst in row[str_name]:
                                curr_obj.add(inst)
                                curr_obj_type[inst] = str(obj)
                    shared_obj = curr_obj.intersection(set([a for b in successor.object.tolist() for a in b]))#set(successor['object']))
                    for o in shared_obj:
                        curr_obj.remove(o)
                    if curr_obj!=set():
                        remaining_obj = ['products', 'orders', 'items', 'employees', 'customers', 'packages']#objects
                        for inst in curr_obj:
                            obj_inst = curr_obj_type[inst]
                            if object_handling[obj_inst]:
                                if obj_inst in remaining_obj:
                                    remaining_obj.remove(obj_inst)
                                    #add high level event of object type leave to log
                                    event.append('object type ' + str(obj_inst) + ' leaves the process in activity ' + str(row['ocel:activity']))
                                    event_type.append('tl')
                                    frequency.append(1)
                                    #TODO: check times
                                    start_time.append(min_time + (curr_seg_count-1)*time_segment)
                                    end_time.append(min_time + curr_seg_count*time_segment)
                                    time_frame.append(curr_seg_count)
                                    qcid.append(case)
                                    related_object.append(obj_inst)
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
                    join_df = successor.copy()
                    split_obj = []
                    for index_j, row_j in join_df.iterrows():
                        join_obj_type = row_j['ocel:type']
                        if object_handling[join_obj_type]:
                            split_obj.append(row_j['ocel:type'])
                        else:
                            split_obj.append(row_j['object'])
                    #joining_obj_types = join_df['ocel:type'].unique()
                    #join_df = join_df.drop_duplicates(subset=['ocel:activity'])
                    #if instance matters for one involved object type, take instances for all object types
                    #handling = True/False and set in loop
                    #for obj in joining_obj_types:
                    #    if object_handling[obj]:
                    #        split_obj.append()
                    #if handling:
                    #add high level event of object type split to log
                    event.append('object (type) split in activity ' + str(row['ocel:activity']) + ' with object (types) ' + str(split_obj))
                    event_type.append('s')
                    frequency.append(len(split_obj))
                    #TODO: check times
                    start_time.append(min_time + (curr_seg_count-1)*time_segment)
                    end_time.append(min_time + curr_seg_count*time_segment) 
                    time_frame.append(curr_seg_count)
                    qcid.append(case)
                    related_object.append(split_obj)
                    #else:
                        #add high level event of object split to log
                    #    event.append('object split in activity ' + str(row['ocel:activity']) + ' with objects ' + str(join_df['object'].sum()))
                    #    event_type.append('s')
                    #    frequency.append(len(join_df['object'].sum()))
                        #TODO: check times
                    #    start_time.append(min_time + (curr_seg_count-1)*time_segment)
                    #    end_time.append(min_time + curr_seg_count*time_segment) 
                    #    time_frame.append(curr_seg_count)
                    #    qcid.append(case)
                    #    related_object.append(join_df['object'].sum())
        for index, row in case_df.iterrows():
            predecessor, case_df = get_predecessor(row, index, case_df, objects)
            if str(predecessor)!='NaN':
                #get execution time and predecessor time and index
                exec_time = row['ocel:timestamp']
                early_time = min(predecessor['ocel:timestamp'])
                early_index = predecessor[predecessor['ocel:timestamp']==early_time].index
                late_time = max(predecessor['ocel:timestamp'])
                late_index = predecessor[predecessor['ocel:timestamp']==early_time].index
                #lagging and delay time are based on opera measures but need to be calculated manually because of different log format
                lagging_time = late_time - early_time
                delay_time = exec_time - late_time
                #add high level event of high lagging time per activity to log
                for i in late_index:
                    lagging_obj_type = predecessor.loc[i, 'ocel:type']
                    lagging_object = predecessor.loc[i, 'object']
                    if object_handling[lagging_obj_type]:
                        #object type
                        event.append('high lagging time of object type' + lagging_obj_type + ' for activity ' + str(row['ocel:activity']))
                        event_type.append('lt')
                        frequency.append(lagging_time)
                        start_time.append(min_time + (curr_seg_count-1)*time_segment)
                        end_time.append(min_time + curr_seg_count*time_segment)
                        time_frame.append(curr_seg_count)
                        qcid.append(case)
                        related_object.append(lagging_obj_type)
                    else:
                        #object instance
                        event.append('high lagging time of object ' + str(lagging_object) + ' for activity ' + str(row['ocel:activity']))
                        event_type.append('lt')
                        frequency.append(lagging_time)
                        start_time.append(min_time + (curr_seg_count-1)*time_segment)
                        end_time.append(min_time + curr_seg_count*time_segment)
                        time_frame.append(curr_seg_count)
                        qcid.append(case)
                        related_object.append(lagging_object)
                #discover joins of object types
                pred_act = set(predecessor['ocel:activity'])
                #test if join happens
                #check if object type joins
                if len(pred_act)==1:
                    #get object types of current event
                    #TODO: change set to list
                    curr_obj = set()
                    curr_obj_type = {}
                    for obj in objects:
                        str_name = 'ocel:type:' + str(obj)
                        if row[str_name]!=[]:
                            for inst in row[str_name]:
                                curr_obj.add(inst)
                                curr_obj_type[inst] = str(obj)
                    shared_obj = curr_obj.intersection(set([a for b in predecessor.object.tolist() for a in b]))
                    for o in shared_obj:
                        curr_obj.remove(o)
                    if curr_obj!=set():
                        remaining_obj = ['products', 'orders', 'items', 'employees', 'customers', 'packages']#objects
                        for inst in curr_obj:
                            obj_inst = curr_obj_type[inst]
                            if object_handling[obj_inst]:
                                if obj_inst in remaining_obj:
                                    remaining_obj.remove(obj_inst)
                                    #add high level event of object type join to log
                                    event.append('object type ' + str(obj_inst) + ' joins the process in activity ' + str(row['ocel:activity']))
                                    event_type.append('tj')
                                    frequency.append(1)
                                    #TODO: check times
                                    start_time.append(min_time + (curr_seg_count-1)*time_segment)
                                    end_time.append(min_time + curr_seg_count*time_segment)
                                    time_frame.append(curr_seg_count)
                                    qcid.append(case)
                                    related_object.append(obj_inst)
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
                    join_df = predecessor.copy()
                    split_obj = []
                    for index_j, row_j in join_df.iterrows():
                        join_obj_type = row_j['ocel:type']
                        if object_handling[join_obj_type]:
                            split_obj.append(row_j['ocel:type'])
                        else:
                            split_obj.append(row_j['object'])
                    #joining_obj_types = join_df['ocel:type'].unique()
                    #join_df = join_df.drop_duplicates(subset=['ocel:activity'])
                    #if instance matters for one involved object type, take instances for all object types
                    #handling = True
                    #for obj in joining_obj_types:
                    #    if object_handling[obj]==False:
                    #        handling = False
                    #if handling:
                        #add high level event of object type join to log
                    event.append('object (type) join in activity ' + str(row['ocel:activity']) + ' with objects ' + str(split_obj))
                    event_type.append('j')
                    frequency.append(len(split_obj))
                    #TODO: check times
                    start_time.append(min_time + (curr_seg_count-1)*time_segment)
                    end_time.append(min_time + curr_seg_count*time_segment)  
                    time_frame.append(curr_seg_count) 
                    qcid.append(case)
                    related_object.append(split_obj)    
                    #else:
                        #add high level event of object join to log
                    #    event.append('object join in activity ' + str(row['ocel:activity']) + ' with objects ' + str(join_df['object'].sum()))
                    #    event_type.append('j')
                    #    frequency.append(len(join_df['object'].sum()))
                    #    #TODO: check times
                    #    start_time.append(min_time + (curr_seg_count-1)*time_segment)
                    #    end_time.append(min_time + curr_seg_count*time_segment)  
                    #    time_frame.append(curr_seg_count) 
                    #    qcid.append(case)
                    #    related_object.append(join_df['object'].sum()) 
              

    #convert to dataframe
    d = {'event': event, 'event type': event_type, 'frequency': frequency, 'start time': start_time, 'end time': end_time, 'time segment': time_frame, 'quasi case id': qcid, 'related object': related_object}
    hl_log = pd.DataFrame(data=d)
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    save_name = parent_path + '/hl_logs_new_new/hl_correction' + str(curr_seg_count) + '.csv'
    hl_log.to_csv(save_name)
    print(hl_log)
    print(curr_seg_count)

def preprocess_case_df(df, objects):
    num_times = len(set(df['ocel:timestamp']))
    if num_times != len(df.index):
        copy_df = df.copy()
        copy_df = copy_df.duplicated(subset=['ocel:timestamp'], keep=False)
        copy_df = copy_df[copy_df]
        c = 0
        equal = True
        for index in copy_df.index:
            if c>0:
                for obj in objects:
                    str_name = 'ocel:type:' + str(obj)
                    if set(df.loc[index, str_name])!=set(df.loc[last_index, str_name]):
                        equal = False
                if equal:
                    df.at[last_index, 'ocel:activity'] = df.loc[last_index, 'ocel:activity'] + ", " + df.loc[index, 'ocel:activity']
                    df.at[last_index, 'ocel:eid'] = df.loc[last_index, 'ocel:eid'] + ", " + df.loc[index, 'ocel:eid']
                    df.drop(index, inplace=True)
                else:
                    last_index = index
                    equal = True
            else:
                last_index = index
            c = c+1
        return df
    else:
        return df

#for event in row at index index and case in df return all direct predecessors in pred and the case without the seen objects in case_df
def get_predecessor(row, index, df, objects):
    save_df = df.copy()
    if row['ocel:timestamp']==min(df['ocel:timestamp']):
        return ('NaN', df)
    else:
        act_time = row['ocel:timestamp']
        df.drop(index, inplace=True)
        copy_df = df.copy()
        for obj in objects:
            str_name = 'intersect:' + str(obj)
            str2 = 'ocel:type:' + str(obj)
            df[str_name] = copy_df[str2].apply(lambda x: list_intersect(x, str2, row))
        time = []
        act = []
        obj_type = []
        obj = []
        for i, r in df.iterrows():
            if r['ocel:timestamp']<act_time:
                for object in objects:
                    str_name = 'intersect:' + str(object)
                    str2 = 'ocel:type:' + str(object)
                    if r[str_name]!=[]:
                        #idea: remove objects from case df row which already intersected an event
                        #problem: object flow can split for one specific object
                        #remark: generell objects should be removed, but if events happen at same time -> problem
                        #solution: merge events with same object and timestamps to one event
                        #next problem: if objects are removed from event, they cant be found in predecessor search
                        #solution: shouldnt be a problem for non-circulat behaviour
                        #comment: indexing safe_df via df index is fine because its copied
                        #TODO: implement
                        new_obj = save_df.loc[i, str2]
                        for element in r[str_name]:
                            if element in new_obj: #if added because of error without caseids
                                new_obj.remove(element)
                        save_df.at[i, str2] = new_obj
                        time.append(r['ocel:timestamp'])
                        act.append(r['ocel:activity'])
                        obj_type.append(str(object))
                        obj.append(r[str_name])
                        #pred = pd.concat([pred, r.to_frame().T])
        if time == []:
            return 'NaN', save_df
        else:
            d = {'ocel:timestamp': time, 'ocel:activity': act, 'ocel:type': obj_type, 'object': obj}
            pred = pd.DataFrame(data=d)
            return pred, save_df
        
def get_successor(row, index, df, objects):
    save_df = df.copy()
    if row['ocel:timestamp']==max(df['ocel:timestamp']): #possible because of preprocessing
        return 'NaN'
    else:
        act_time = row['ocel:timestamp']
        act_objects = []
        for obj in objects:
            str_name = 'ocel:type:' + str(obj)
            act_objects.extend(row[str_name])
        save_df.drop(index, inplace=True)
        copy_df = df.copy()
        for obj in objects:
            str_name = 'intersect:' + str(obj)
            str2 = 'ocel:type:' + str(obj)
            save_df[str_name] = copy_df[str2].apply(lambda x: list_intersect(x, str2, row))
        time = []
        act = []
        obj_type = []
        obj = []
        #idea: df is sorted, therefore first successor of one object is indeed sucessor but objects do not need to be modified because found object is removed from current activity object list
        for i, r in save_df.iterrows():
            #if should be unnecessary
            if r['ocel:timestamp']>act_time:
                for object in objects:
                    str_name = 'intersect:' + str(object)
                    if r[str_name]!=[]:
                        is_succ = False
                        for element in r[str_name]:
                            is_in = element in act_objects
                            if is_in:
                                is_succ = True
                                act_objects.remove(element) 
                        if is_succ:
                            time.append(r['ocel:timestamp'])
                            act.append(r['ocel:activity'])
                            obj_type.append(str(object))
                            obj.append(r[str_name])
        if time == []:
            return 'NaN'
        else:
            d = {'ocel:timestamp': time, 'ocel:activity': act, 'ocel:type': obj_type, 'object': obj}
            succ = pd.DataFrame(data=d)
            return succ

def search_patterns(df_in, objects, curr_act):
    df = df_in.copy()
    print('df begin search pattern:')
    print(df)
    #keep first and last occurences of object patterns 
    #idea: see if patterns changes over time
    #TODO additionally: search for deviations
    object_list = ['ocel:type:material', 'ocel:type:purchase_order']
    curr_obj = ['material', 'purchase_order']
    #for obj in objects:
    #    ocel = 'ocel:type:' + str(obj)
    #    object_list.append(ocel)
    print(object_list)
    pattern_firsts = df.loc[df.astype(str).drop_duplicates(subset=object_list, keep='first').index]
    pattern_lasts = df.loc[df.astype(str).drop_duplicates(subset=object_list, keep='last').index]
    print(pattern_firsts.shape)
    print(pattern_lasts.shape)

    pattern = []
    start = []
    end = []
    num = 0
    for index, row in pattern_firsts.iterrows():
        num = num + 1
        matched = False
        for i, r in pattern_lasts.iterrows():
            equal = True
            for obj in curr_obj:
                str_name = 'ocel:type:' + str(obj)
                if row[str_name] != r[str_name]:
                    equal = False
            if equal:  
                start.append(row['ocel:timestamp'])
                end.append(r['ocel:timestamp'])
                pattern.append('pattern ' + str(num))
                pattern_lasts.drop(i, inplace=True)
                matched = True
        if not matched:
            #should not happen
            start.append(row['ocel:timestamp'])
            end.append(row['ocel:timestamp'])
            pattern.append('pattern ' + str(num))
            print('not matched')
            print(row)

    print('lasts left:')
    print(pattern_lasts.shape)
    timeline_df = pd.DataFrame({'pattern': pattern, 'start': start, 'end': end})
    print(timeline_df)

    
    #pattern_firsts = df.drop_duplicates(subset=object_list, keep='first')
    #pattern_lasts = df.drop_duplicates(subset=object_list, keep='last')

    print(pattern_firsts)
    print(pattern_lasts.to_string())
    
    timeline_df['start'] = pd.to_datetime(timeline_df['start'])
    timeline_df['end'] = pd.to_datetime(timeline_df['end'])
    timeline_df['diff'] = timeline_df['end'] - timeline_df['start']
    
    # Declaring a figure "gnt"
    fig, gnt = plt.subplots(figsize=(8,6))    
    y_tick_labels = timeline_df.pattern.values
    y_pos = np.arange(len(y_tick_labels))
    gnt.set_yticks(y_pos)
    gnt.set_yticklabels(y_tick_labels)
    for index, row in timeline_df.iterrows():
        start_year = int(row.start.strftime("%Y"))
        duration = row['diff'].days/365
        gnt.broken_barh([(start_year-0.01, duration+0.01)], 
                        (index,0.2), 
                        facecolors =('tan'))
        #gnt.text(start_year+0.5, index-0.2, row.pattern)

    save_name = str(curr_act) + '_timeline.png'
    plt.savefig(save_name)

    #plot_df = pd.DataFrame()
    #plot_df['start'] = pattern_firsts
    #for index, row in pattern_firsts.iterrows():
    #    for i, r in pattern_lasts.iterrows():
    #        print

    #observation: in this dataset, only the number of involved materials really changes (invoice in two activites)
    #this is not really a deviation or a change of pattern over time
    #idea: maybe in a next step compute how many object types vary and set a threshold for deviations/patterns
    #also: there could be a pattern within the deviation, e.g in this dataset the different numbers of involved materials are often identical    
    

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

#def quasi_case_id(df, objects):
#    object_dict = {}
#    case_num = 0
#    for index, row in df.iterrows():
#        for obj in objects:
#            str_name = 'ocel:type' + str(obj)
#            for inst in df[str_name]:
#                inst_case = object_dict.get(inst)
#                if inst_case == None:
#                    case_num = case_num+1
#                    object_dict[inst_case] = case_num
#                else:
#                    curr
def pattern_detection():
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
    for i in range(105):
        str_name = parent_path + '/hl_logs_new/hl_log' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)

    df['id'] = df.index
    copy = df.copy()
    print(copy.head(n=10))

    #filter high load events by just keeping the highest loads per object type (above mittelwert)
    df = copy.copy()
    df_hl = df[df['event type']=='hl']
    df_hl = df_hl.astype({'frequency':'int'})
    filtered_df_hl = pd.DataFrame()
    for obj in objects:
        df_hl_obj = df_hl[df_hl['related object']==obj]
        if len(df_hl_obj['frequency'])>0:
            max_obj = int(max(df_hl_obj['frequency']))
        filtered = df_hl_obj[df_hl_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))]
        filtered_df_hl = pd.concat([filtered_df_hl, filtered])
    #print(filtered_df_hl)
        
    #filter high demand events by just keeping the highest demands per object
    df = copy.copy()
    df_hd = df[df['event type']=='hd']
    df_hd = df_hd.astype({'frequency':'int'})
    filtered_df_hd = df_hd.sort_values('frequency', ascending=False)
    num_hd = len(df_hd)
    filtered_df_hd = filtered_df_hd.head(n=int(num_hd/10))
    #print(filtered_df_hd)

    #filter high overall load events by just keeping the above median freuquent ones
    df = copy.copy()
    df_hol = df[df['event type']=='hol']
    df_hol = df_hol.astype({'frequency':'int'})
    df_hol = df_hol.sort_values(by=['frequency'])
    half = int(len(df_hol)/2)
    df_hol.drop(df_hol.head(half).index,inplace=True)
    print(df_hol)

    #filter object type leaves process events
    df = copy.copy()
    df_tl = df[df['event type']=='tl']
    #print(df_tl)

    #filter object type joins process events
    df = copy.copy()
    df_tj = df[df['event type']=='tj']
    #print(df_tj)

    #filter split of object types events
    df = copy.copy()
    df_s = df[df['event type']=='s']
    #print(df_s)

    #filter join of object type events
    df = copy.copy()
    df_j = df[df['event type']=='j']
    #print('len of join df: ' + str(len(df_j)))
    #print(df_j)

    #filter lagging time events by keeping the highest 10%
    df = copy.copy()
    df_lt = df[df['event type']=='lt']
    df_lt.sort_values('frequency', ascending=False, inplace=True)
    number_lt = len(df_lt)
    filtered_lt = df_lt.head(n=int(number_lt/10))
    #print(filtered_lt)

    #filter lagging time events by keeping the lowest 10%
    df = copy.copy()
    df_llt = df[df['event type']=='lt']
    df_llt.sort_values('frequency', ascending=True, inplace=True)
    number_llt = len(df_llt)
    filtered_llt = df_llt.head(n=int(number_llt/10))

    #look for patterns
    #initialize lists for relations
    #direct realtions
    direct_act1 = []
    direct_act2 = []
    direct_id1 = []
    direct_id2 = []
    direct_time1 = []
    direct_time2 = []
    #not direct realtions
    not_direct_act1 = []
    not_direct_act2 = []
    not_direct_id1 = []
    not_direct_id2 = []
    not_direct_time1 = []
    not_direct_time2 = []
    #complete direct conditional relation
    direct_con1 = []
    direct_con2 = []
    direct_con3 = []
    direct_con_id1 = []
    direct_con_id2 = []
    direct_con_id3 = []
    direct_con_time1 = []
    direct_con_time2 = []
    direct_con_time3 = []
    #not complete direct conditianl relation
    not_direct_con1 = []
    not_direct_con2 = []
    not_direct_con3 = []
    not_direct_con_id1 = []
    not_direct_con_id2 = []
    not_direct_con_id3 = []
    not_direct_con_time1 = []
    not_direct_con_time2 = []
    not_direct_con_time3 = []
    #lagging time focused relations
    filtered_lt.sort_values('time segment', ascending=False, inplace=True)
    for index, row in filtered_lt.iterrows():
        curr_time_seg = row['time segment']
        curr_obj = row['related object']
        if curr_obj in objects:
            #look for relation hl->lt
            tmp_df = filtered_df_hl.copy()
            tmp_df.sort_values('time segment')# not really necessary
            tmp_df = tmp_df[tmp_df['time segment']<curr_time_seg]
            for i, r in tmp_df.iterrows():
                rel_obj = r['related object']
                if rel_obj==curr_obj:
                    if r['time segment'] == curr_time_seg - 1:
                        direct_act1.append(r['event'])
                        direct_time1.append(r['time segment'])
                        direct_id1.append(r['id'])
                        direct_act2.append(row['event'])
                        direct_time2.append(row['time segment'])
                        direct_id2.append(row['id'])
                    else:
                        not_direct_act1.append(r['event'])
                        not_direct_time1.append(r['time segment'])
                        not_direct_id1.append(r['id'])
                        not_direct_act2.append(row['event'])
                        not_direct_time2.append(row['time segment'])
                        not_direct_id2.append(row['id'])
            #look for relation s->lt with object types
            tmp_df = df_s.copy()
            tmp_df.sort_values('time segment')# not really necessary
            tmp_df = tmp_df[tmp_df['time segment']<curr_time_seg]
            for i, r in tmp_df.iterrows():
                rel_obj = r['related object']
                #workaround because event is not properly implemented yet
                #TODO: implement split event such that object type gets listed
                if curr_obj=='items':
                    if rel_obj.find('i-')>0 or rel_obj.find('items')>0:
                        if r['time segment'] == curr_time_seg - 1:
                            direct_act1.append(r['event'])
                            direct_time1.append(r['time segment'])
                            direct_id1.append(r['id'])
                            direct_act2.append(row['event'])
                            direct_time2.append(row['time segment'])
                            direct_id2.append(row['id'])
                        else:
                            not_direct_act1.append(r['event'])
                            not_direct_time1.append(r['time segment'])
                            not_direct_id1.append(r['id'])
                            not_direct_act2.append(row['event'])
                            not_direct_time2.append(row['time segment'])
                            not_direct_id2.append(row['id'])
                if curr_obj=='orders':
                    if rel_obj.find('o-')>0 or rel_obj.find('orders')>0:
                        if r['time segment'] == curr_time_seg - 1:
                            direct_time1.append(r['time segment'])
                            direct_id1.append(r['id'])
                            direct_act1.append(r['event'])
                            direct_act2.append(row['event'])
                            direct_time2.append(row['time segment'])
                            direct_id2.append(row['id'])
                        else:
                            not_direct_time1.append(r['time segment'])
                            not_direct_id1.append(r['id'])
                            not_direct_act1.append(r['event'])
                            not_direct_act2.append(row['event'])
                            not_direct_time2.append(row['time segment'])
                            not_direct_id2.append(row['id'])
                if curr_obj=='packages':
                    if rel_obj.find('p-')>0 or rel_obj.find('packages')>0:
                        if rel_obj==curr_obj:
                            if r['time segment'] == curr_time_seg - 1:
                                direct_time1.append(r['time segment'])
                                direct_id1.append(r['id'])
                                direct_act1.append(r['event'])
                                direct_act2.append(row['event'])
                                direct_time2.append(row['time segment'])
                                direct_id2.append(row['id'])
                            else:
                                not_direct_time1.append(r['time segment'])
                                not_direct_id1.append(r['id'])
                                not_direct_act1.append(r['event'])
                                not_direct_act2.append(row['event'])
                                not_direct_time2.append(row['time segment'])
                                not_direct_id2.append(row['id'])
            #look for relation j->lt with object types and hl->j->lt
            tmp_df = df_j.copy()
            tmp_df.sort_values('time segment')# not really necessary
            tmp_df = tmp_df[tmp_df['time segment']<curr_time_seg]
            for i, r in tmp_df.iterrows():
                first_cond = False
                rel_obj = r['related object']
                #workaround because event is not properly implemented yet
                #TODO: implement split event such that object type gets listed
                if curr_obj=='items':
                    if rel_obj.find('i-')>0 or rel_obj.find('items')>0:
                        first_cond = True
                        mid_time = r['time segment']
                        second_event = r['event']
                        third_event = row['event']
                        if r['time segment'] == curr_time_seg - 1:
                            direct_time1.append(r['time segment'])
                            direct_id1.append(r['id'])
                            direct_act1.append(r['event'])
                            direct_act2.append(row['event'])
                            direct_time2.append(row['time segment'])
                            direct_id2.append(row['id'])
                        else:
                            not_direct_time1.append(r['time segment'])
                            not_direct_id1.append(r['id'])
                            not_direct_act1.append(r['event'])
                            not_direct_act2.append(row['event'])
                            not_direct_time2.append(row['time segment'])
                            not_direct_id2.append(row['id'])
                if curr_obj=='orders':
                    if rel_obj.find('o-')>0 or rel_obj.find('orders')>0:
                        first_cond = True
                        mid_time = r['time segment']
                        second_event = r['event']
                        third_event = row['event']
                        if r['time segment'] == curr_time_seg - 1:
                            direct_time1.append(r['time segment'])
                            direct_id1.append(r['id'])
                            direct_act1.append(r['event'])
                            direct_act2.append(row['event'])
                            direct_time2.append(row['time segment'])
                            direct_id2.append(row['id'])
                        else:
                            not_direct_time1.append(r['time segment'])
                            not_direct_id1.append(r['id'])
                            not_direct_act1.append(r['event'])
                            not_direct_act2.append(row['event'])
                            not_direct_time2.append(row['time segment'])
                            not_direct_id2.append(row['id'])
                if curr_obj=='packages':
                    if rel_obj.find('p-')>0 or rel_obj.find('packages')>0:
                        first_cond = True
                        mid_time = r['time segment']
                        second_event = r['event']
                        third_event = row['event']
                        if rel_obj==curr_obj:
                            if r['time segment'] == curr_time_seg - 1:
                                direct_act1.append(r['event'])
                                direct_time1.append(r['time segment'])
                                direct_id1.append(r['id'])
                                direct_act2.append(row['event'])
                                direct_time2.append(row['time segment'])
                                direct_id2.append(row['id'])
                            else:
                                not_direct_act1.append(r['event'])
                                not_direct_time1.append(r['time segment'])
                                not_direct_id1.append(r['id'])
                                not_direct_act2.append(row['event'])
                                not_direct_time2.append(row['time segment'])
                                not_direct_id2.append(row['id'])
                if first_cond:
                    tmp_df2 = filtered_df_hl.copy()
                    tmp_df2.sort_values('time segment')# not really necessary
                    tmp_df2 = tmp_df2[tmp_df2['time segment']<mid_time]
                    for ind, ro in tmp_df2.iterrows():
                        first_obj = ro['related object']
                        if first_obj==curr_obj:
                            first_event = ro['event']
                            if mid_time==curr_time_seg-1 and ro['time segment']==mid_time-1:
                                direct_con1.append(first_event)
                                direct_con_time1.append(ro['time segment'])
                                direct_con_id1.append(ro['id'])
                                direct_con2.append(second_event)
                                direct_con_time2.append(r['time segment'])
                                direct_con_id2.append(r['id'])
                                direct_con3.append(third_event)
                                direct_con_time3.append(row['time segment'])
                                direct_con_id3(row['id'])
                            else:
                                not_direct_con1.append(first_event)
                                not_direct_con_time1.append(ro['time segment'])
                                not_direct_con_id1.append(ro['id'])
                                not_direct_con2.append(second_event)
                                not_direct_con_time2.append(r['time segment'])
                                not_direct_con_id2.append(r['id'])
                                not_direct_con3.append(third_event)
                                not_direct_con_time3.append(row['time segment'])
                                not_direct_con_id3(row['id'])
        else:
            #look for relation hd->lt
            tmp_df = filtered_df_hd.copy()
            tmp_df.sort_values('time segment')# not really necessary
            tmp_df = tmp_df[tmp_df['time segment']<curr_time_seg]
            for i, r in tmp_df.iterrows():
                rel_obj = r['related object']
                if curr_obj.find(rel_obj)>0:
                    if r['time segment'] == curr_time_seg - 1:
                        direct_act1.append(r['event'])
                        direct_time1.append(r['time segment'])
                        direct_id1.append(r['id'])
                        direct_act2.append(row['event'])
                        direct_time2.append(row['time segment'])
                        direct_id2.append(row['id'])
                    else:
                        not_direct_act1.append(r['event'])
                        not_direct_time1.append(r['time segment'])
                        not_direct_id1.append(r['id'])
                        not_direct_act2.append(row['event'])
                        not_direct_time2.append(row['time segment'])
                        not_direct_id2.append(row['id'])
            #look for relation s->lt for object instance
            tmp_df = df_s.copy()
            tmp_df.sort_values('time segment')# not really necessary
            tmp_df = tmp_df[tmp_df['time segment']<curr_time_seg]
            for i, r in tmp_df.iterrows():
                rel_obj = r['related object']
                if rel_obj.find(curr_obj)>0:
                    if r['time segment'] == curr_time_seg - 1:
                        direct_act1.append(r['event'])
                        direct_time1.append(r['time segment'])
                        direct_id1.append(r['id'])
                        direct_act2.append(row['event'])
                        direct_time2.append(row['time segment'])
                        direct_id2.append(row['id'])
                    else:
                        not_direct_act1.append(r['event'])
                        not_direct_time1.append(r['time segment'])
                        not_direct_id1.append(r['id'])
                        not_direct_act2.append(row['event'])
                        not_direct_time2.append(row['time segment'])
                        not_direct_id2.append(row['id'])
            #look for relation j->lt and hd->j->lt
            tmp_df = df_j.copy()
            tmp_df.sort_values('time segment')# not really necessary
            tmp_df = tmp_df[tmp_df['time segment']<curr_time_seg]
            for i, r in tmp_df.iterrows():
                rel_obj = r['related object']
                if rel_obj.find(curr_obj)>0:
                    if r['time segment'] == curr_time_seg - 1:
                        direct_act1.append(r['event'])
                        direct_time1.append(r['time segment'])
                        direct_id1.append(r['id'])
                        direct_act2.append(row['event'])
                        direct_time2.append(row['time segment'])
                        direct_id2.append(row['id'])
                    else:
                        not_direct_act1.append(r['event'])
                        not_direct_time1.append(r['time segment'])
                        not_direct_id1.append(r['id'])
                        not_direct_act2.append(row['event'])
                        not_direct_time2.append(row['time segment'])
                        not_direct_id2.append(row['id'])
                    tmp_df2 = filtered_df_hd.copy()
                    tmp_df2.sort_values('time segment')# not really necessary
                    tmp_df2 = tmp_df2[tmp_df2['time segment']<mid_time]  
                    for ind, ro in tmp_df2.iterrows():
                        third_obj = r['related object']
                        if third_obj.find(curr_obj)>0:
                            if r['time segment']==curr_time_seg-1 and ro['time segment']==r['time segment']-1:
                                direct_con1.append(first_event)
                                direct_con_time1.append(ro['time segment'])
                                direct_con_id1.append(ro['id'])
                                direct_con2.append(second_event)
                                direct_con_time2.append(r['time segment'])
                                direct_con_id2.append(r['id'])
                                direct_con3.append(third_event)
                                direct_con_time3.append(row['time segment'])
                                direct_con_id3(row['id'])
                            else:
                                not_direct_con1.append(first_event)
                                not_direct_con_time1.append(ro['time segment'])
                                not_direct_con_id1.append(ro['id'])
                                not_direct_con2.append(second_event)
                                not_direct_con_time2.append(r['time segment'])
                                not_direct_con_id2.append(r['id'])
                                not_direct_con3.append(third_event)
                                not_direct_con_time3.append(row['time segment'])
                                not_direct_con_id3(row['id'])

    #object occurence based relations
    df_hol.sort_values('time segment', ascending=False, inplace=True)
    for index, row in df_hol.iterrows():
        print('test')

    d = {'event1': not_direct_act1, 'id1': not_direct_id1, 'time1': not_direct_time1, 'event2': not_direct_act2, 'id2': not_direct_id2, 'time2': not_direct_time2}
    hl_log = pd.DataFrame(data=d)
    print(hl_log[5:91])
    #for i in range(len(not_direct_act1)):
        #text = not_direct_act1[i] + '->' + not_direct_act2[i]
        #print(text)
    print('len of cond: ' + str(len(not_direct_con1)))
    for i in range(len(not_direct_con1)):
        text = '(' + not_direct_con1[i] + '->' + not_direct_con2[i] + ')->' + not_direct_con3[i]
        #print(text)
    #df.to_csv('hl_test.csv')

def pattern_detection_test():
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
        str_name = parent_path + '/hl_logs_new_new/hl_log' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)
        str_name = parent_path + '/hl_logs_new_new/hl_correction' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)

    df['id'] = df.index
    print('number of events:')
    print(len(df))
    print('really all events: 87162-427(hol)-12(hl)-1336(hd)=85387')
    print(df['time segment'].unique())

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
        print(obj)
        print(len(df_hl_obj))
        if len(df_hl_obj['frequency'])>0:
            max_obj = int(max(df_hl_obj['frequency']))
        df_hl_obj['frequency'] = df_hl_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))
        filtered = df_hl_obj[df_hl_obj['frequency']==True]
        #filtered = df_hl_obj[df_hl_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))]
        filtered_df_hl = pd.concat([filtered_df_hl, filtered])
    print('high load events after filtering:')
    print(len(filtered_df_hl))
    #print(filtered_df_hl)
        
    #filter high demand events by just keeping the highest demands per object
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

    new_df_case = pd.concat([df_l, df_j, filtered_llt, filtered_lt], ignore_index=True)
    new_df_obj = pd.concat([filtered_df_hl, filtered_df_hd, filtered_llt, filtered_lt], ignore_index=True)#df_s, df_m, df_l, df_j,
    new_df = pd.concat([filtered_df_hl, filtered_df_hd, df_s, df_m, df_l, df_j, filtered_llt, filtered_lt], ignore_index=True)#len=16230, ohne s,m,l,j: len=1231
    print(len(new_df_obj))

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
    result_df = pd.DataFrame()
    case_ids = []
    count = 0
    #look for patterns that are based on events with the same object
    for obj in all_objects:
        print(obj)
        df_obj = new_df.copy()
        df_obj['related object'] = df_obj['related object'].apply(lambda x: x if obj in x else np.nan)
        df_obj.dropna(subset=['related object'], inplace=True)
        df_obj2 = df_obj.copy()
        for index, row in df_obj.iterrows():
            for i, r in df_obj2.iterrows():
                if row['time segment']==r['time segment']-1:
                    #check for hd->hlt, hl->hlt, s->hlt, m->hlt
                    if (row['event type']=='hd' and r['event type']=='hlt') or (row['event type']=='hl' and r['event type']=='hlt') or (row['event type']=='s' and r['event type']=='hlt'):
                        result_df = pd.concat([result_df, row.to_frame().T, r.to_frame().T])
                        #result_df.loc[index, 'case id'] = count
                        #result_df.loc[i, 'case id'] = count
                        case_ids.append(count)
                        case_ids.append(count)
                        count = count + 1
                    if (row['event type']=='m' and r['event type']=='hlt'):
                        result_df = pd.concat([result_df, row.to_frame().T, r.to_frame().T])
                        case_ids.append(count)
                        case_ids.append(count)
                        count = count + 1
                        df_obj3 = df_obj.copy()
                        for ind, ro in df_obj3.iterrows():
                            if (ro['time segment']==row['time segment']-1) and (ro['event type']=='hd' or ro['event type']=='hl'):
                                result_df = pd.concat([result_df, ro.to_frame().T, row.to_frame().T, r.to_frame().T])
                                case_ids.append(count)
                                case_ids.append(count)
                                case_ids.append(count)
                                count = count + 1
    print(result_df)
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    save_name = parent_path + '/result_df0.csv'
    print(save_name)
    result_df.to_csv(save_name)
    #result df size at this point: 2024635 entries
    #look for patterns that are based on events within the same process execution
    cases = set(new_df['quasi case id'])
    for case in cases:
        print(case)
        df_case = new_df_case.copy()
        df_case = df_case[df_case['quasi case id']==case]
        df_case2 = df_case.copy()
        for index, row in df_case.iterrows():
            for i, r in df_case2.iterrows():
                if row['time segment']==r['time segment']-1:
                    #check for j->hlt
                    if (row['event type']=='j' and r['event type']=='hlt'):
                        result_df = pd.concat([result_df, row.to_frame().T, r.to_frame().T])
                        case_ids.append(count)
                        case_ids.append(count)
                        count = count + 1
                    #check for j->j to construct (j->)* hlt later
                    if (row['event type']=='j' and r['event type']=='j'):
                        result_df = pd.concat([result_df, row.to_frame().T, r.to_frame().T])
                        case_ids.append(count)
                        case_ids.append(count)
                        count = count + 1
                    #check for l->l to to construct (l->)* llt later
                    if (row['event type']=='l' and r['event type']=='l'):
                        result_df = pd.concat([result_df, row.to_frame().T, r.to_frame().T])
                        case_ids.append(count)
                        case_ids.append(count)
                        count = count + 1
                    #check for l->llt, hlt->l->llt
                    if (row['event type']=='l' and r['event type']=='llt'):
                        result_df = pd.concat([result_df, row.to_frame().T, r.to_frame().T])
                        case_ids.append(count)
                        case_ids.append(count)
                        count = count + 1
                        df_case3 = df_case.copy()
                        for ind, ro in df_case3.iterrows():
                            if (ro['time segment']==row['time segment']-1) and (ro['event type']=='hlt'):
                                result_df = pd.concat([result_df, ro.to_frame().T, row.to_frame().T, r.to_frame().T])
                                case_ids.append(count)
                                case_ids.append(count)
                                case_ids.append(count)
                                count = count + 1
                    #check for s->s to to construct (s->)* llt later
                    if (row['event type']=='s' and r['event type']=='s'):
                        result_df = pd.concat([result_df, row.to_frame().T, r.to_frame().T])
                        case_ids.append(count)
                        case_ids.append(count)
                        count = count + 1
                    #check for s->llt, hlt->s->llt
                    if (row['event type']=='s' and r['event type']=='llt'):
                        result_df = pd.concat([result_df, row.to_frame().T, r.to_frame().T])
                        case_ids.append(count)
                        case_ids.append(count)
                        count = count + 1
                        df_case3 = df_case.copy()
                        for ind, ro in df_case3.iterrows():
                            if (ro['time segment']==row['time segment']-1) and (ro['event type']=='hlt'):
                                lagging_obj = ro['related object'][0]
                                split_obj = row['related object']
                                print('found hlt->s->llt')
                                print(lagging_obj)
                                print(split_obj)
                                if lagging_obj in split_obj:
                                    result_df = pd.concat([result_df, ro.to_frame().T, row.to_frame().T, r.to_frame().T])
                                    case_ids.append(count)
                                    case_ids.append(count)
                                    case_ids.append(count)
                                    count = count + 1
        
    result_df['case id'] = case_ids
    print(result_df)
    save_name = parent_path + '/result_df1.csv'
    result_df.to_csv(save_name)

def pattern_detection_dict():
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
        str_name = parent_path + '/hl_logs_new_new/hl_log' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)
        str_name = parent_path + '/hl_logs_new_new/hl_correction' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)

    df['id'] = df.index
    print('number of events:')
    print(len(df))
    print('really all events: 87162-427(hol)-12(hl)-1336(hd)=85387')
    print(df['time segment'].unique())

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
    #df_hl = df_hl.astype({'frequency':'int'})
    #filtered_df_hl = pd.DataFrame()
    #for obj in objects:
    #    df_hl_obj = df_hl.copy()
    #    df_hl_obj['is related object'] = df_hl_obj['related object'].apply(lambda x: obj in x)
    #    df_hl_obj = df_hl_obj[df_hl_obj['is related object']==True]
    #    df_hl_obj.drop(columns='is related object')
    #    print(obj)
    #    print(len(df_hl_obj))
    #    if len(df_hl_obj['frequency'])>0:
    #        max_obj = int(max(df_hl_obj['frequency']))
    #    df_hl_obj['frequency'] = df_hl_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))
    #    filtered = df_hl_obj[df_hl_obj['frequency']==True]
        #filtered = df_hl_obj[df_hl_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))]
    #    filtered_df_hl = pd.concat([filtered_df_hl, filtered])
    filtered_df_hl = df_hl.copy()
    print('high load events after filtering:')
    print(len(filtered_df_hl))
    #print(filtered_df_hl)
        
    #filter high demand events by just keeping the highest demands per object
    df_hd = df.copy()
    df_hd = df_hd[df_hd['event type']=='hd']
    print(len(df_hd))
    df_hd.drop_duplicates(subset=['event', 'time segment'], inplace=True, keep='first')
    print('all high demand events:')
    print(len(df_hd))
    #df_hd = df_hd.astype({'frequency':'int'})
    #filtered_df_hd = df_hd.sort_values('frequency', ascending=False)
    #num_hd = len(df_hd)
    #filtered_df_hd = filtered_df_hd.head(n=int(num_hd/2))
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

    new_df_case = pd.concat([df_l, df_j, filtered_llt, filtered_lt], ignore_index=True)
    new_df_obj = pd.concat([filtered_df_hl, filtered_df_hd, filtered_llt, filtered_lt], ignore_index=True)#df_s, df_m, df_l, df_j,
    new_df = pd.concat([filtered_df_hl, filtered_df_hd, df_s, df_m, df_l, df_j, filtered_llt, filtered_lt], ignore_index=True)#len=16230, ohne s,m,l,j: len=1231
    print(len(new_df_obj))

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
    all_ids = new_df['id'].unique()
    for id in all_ids:
        result_dict.setdefault(id, [])
    #look for patterns that are based on events with the same object
    for obj in all_objects:
        print(obj)
        df_obj = new_df.copy()
        df_obj['related object'] = df_obj['related object'].apply(lambda x: x if obj in x else np.nan)
        df_obj.dropna(subset=['related object'], inplace=True)
        df_obj2 = df_obj.copy()
        for index, row in df_obj.iterrows():
            for i, r in df_obj2.iterrows():
                if row['time segment']==r['time segment']-1:
                    #check for hd->hlt, hl->hlt, s->hlt, m->hlt
                    if (row['event type']=='hd' and r['event type']=='hlt') or (row['event type']=='hl' and r['event type']=='hlt') or (row['event type']=='s' and r['event type']=='hlt'):
                        first_id = row['id']
                        second_id = r['id']
                        succ_list = result_dict[first_id]
                        succ_list.append(second_id)
                        result_dict[first_id] = succ_list
                    if (row['event type']=='m' and r['event type']=='hlt'):
                        first_id = row['id']
                        second_id = r['id']
                        succ_list = result_dict[first_id]
                        succ_list.append(second_id)
                        result_dict[first_id] = succ_list
                        df_obj3 = df_obj.copy()
                        for ind, ro in df_obj3.iterrows():
                            if (ro['time segment']==row['time segment']-1) and (ro['event type']=='hd' or ro['event type']=='hl'):
                                first_id = ro['id']
                                second_id = row['id']
                                succ_list = result_dict[first_id]
                                succ_list.append(second_id)
                                result_dict[first_id] = succ_list
    print(result_dict)
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    save_name = parent_path + '/result.txt'
    f = open(save_name, "a")
    f.write(result_dict)
    f.close()
    print(save_name)
    #result df size at this point: 2024635 entries
    #look for patterns that are based on events within the same process execution
    cascade_dict = []
    for id in all_ids:
        cascade_dict.setdefault(id, [])
    cases = set(new_df['quasi case id'])
    for case in cases:
        print(case)
        df_case = new_df_case.copy()
        df_case = df_case[df_case['quasi case id']==case]
        df_case2 = df_case.copy()
        for index, row in df_case.iterrows():
            for i, r in df_case2.iterrows():
                if row['time segment']==r['time segment']-1:
                    #check for j->hlt
                    if (row['event type']=='j' and r['event type']=='hlt'):
                        first_id = row['id']
                        second_id = r['id']
                        succ_list = result_dict[first_id]
                        succ_list.append(second_id)
                        result_dict[first_id] = succ_list
                    #check for j->j to construct (j->)* hlt later
                    if (row['event type']=='j' and r['event type']=='j'):
                        first_id = row['id']
                        second_id = r['id']
                        succ_list = cascade_dict[first_id]
                        succ_list.append(second_id)
                        cascade_dict[first_id] = succ_list
                    #check for l->l to to construct (l->)* llt later
                    if (row['event type']=='l' and r['event type']=='l'):
                        first_id = row['id']
                        second_id = r['id']
                        succ_list = cascade_dict[first_id]
                        succ_list.append(second_id)
                        cascade_dict[first_id] = succ_list
                    #check for l->llt, hlt->l->llt
                    if (row['event type']=='l' and r['event type']=='llt'):
                        first_id = row['id']
                        second_id = r['id']
                        succ_list = result_dict[first_id]
                        succ_list.append(second_id)
                        result_dict[first_id] = succ_list
                        df_case3 = df_case.copy()
                        for ind, ro in df_case3.iterrows():
                            if (ro['time segment']==row['time segment']-1) and (ro['event type']=='hlt'):
                                first_id = ro['id']
                                second_id = row['id']
                                succ_list = result_dict[first_id]
                                succ_list.append(second_id)
                                result_dict[first_id] = succ_list
                    #check for s->s to to construct (s->)* llt later
                    if (row['event type']=='s' and r['event type']=='s'):
                        first_id = row['id']
                        second_id = r['id']
                        succ_list = cascade_dict[first_id]
                        succ_list.append(second_id)
                        cascade_dict[first_id] = succ_list
                    #check for s->llt, hlt->s->llt
                    if (row['event type']=='s' and r['event type']=='llt'):
                        first_id = row['id']
                        second_id = r['id']
                        succ_list = result_dict[first_id]
                        succ_list.append(second_id)
                        result_dict[first_id] = succ_list
                        df_case3 = df_case.copy()
                        for ind, ro in df_case3.iterrows():
                            if (ro['time segment']==row['time segment']-1) and (ro['event type']=='hlt'):
                                lagging_obj = ro['related object'][0]
                                split_obj = row['related object']
                                print('found hlt->s->llt')
                                print(lagging_obj)
                                print(split_obj)
                                if lagging_obj in split_obj:
                                    first_id = ro['id']
                                    second_id = row['id']
                                    succ_list = result_dict[first_id]
                                    succ_list.append(second_id)
                                    result_dict[first_id] = succ_list
        
    #result_df['case id'] = case_ids
    print(result_dict)
    print(cascade_dict)
    save_name = parent_path + '/result.txt'
    f = open(save_name, "a")
    f.write(result_dict)
    f.close()

def case_construction():
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '/result_df1.csv'
    result_df = pd.read_csv(str_name)
    print(result_df)
    print(len(result_df['case id'].unique()))

    duplicates = result_df.duplicated(subset=['id'], keep=False)
    print(duplicates)
    d = {'dup': duplicates, 'id': result_df['id']}
    df_dup = pd.DataFrame(data=d)
    print(df_dup)
    df_dup['dup'] = df_dup['dup'].apply(lambda x: x if x else np.nan)
    df_dup.dropna(subset=['dup'], inplace=True)
    print(df_dup)
    dup_ids = set()
    for index, row in df_dup.iterrows():
        #print(row)
        dup_ids.add(row['id'])
    
    result_df['path'] = result_df['case id']
    for id in dup_ids:
        tmp_df = result_df.copy()
        tmp_df = tmp_df[tmp_df['id']==id]
        case_ids = set(tmp_df['case id'])
        replace_dict = dict()
        new_id = min(case_ids)
        #if (len(tmp_df.index)>2):
        #    print(case_ids)
        #    print(new_id)
        for cid in case_ids:
            replace_dict[cid] = new_id
        result_df['case id'].replace(to_replace=replace_dict, inplace=True)
    
    #result_df['member'] = result_df['case id']
    print(result_df)
    print(len(result_df['case id'].unique()))
    print(result_df['case id'].unique())
    for id in result_df['case id'].unique():
        tmp_df = result_df[result_df['case id']==id]
        event_types = tmp_df['event type'].unique()
        if 'hd' in event_types:# and 's' in event_types and 'hlt' in event_types:
            print(tmp_df[['event type', 'quasi case id', 'related object', 'id', 'case id', 'path']])

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
        str_name = parent_path + '/hl_logs_new_new/hl_log' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)
        str_name = parent_path + '/hl_logs_new_new/hl_correction' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)

    df['id'] = df.index
    print('number of events:')
    print(len(df))
    print(df['time segment'].unique())

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
    #print(df_hl)
    #df_hl = df_hl.astype({'frequency':'int'})
    #filtered_df_hl = pd.DataFrame()
    #for obj in objects:
    #    df_hl_obj = df_hl.copy()
    #    df_hl_obj['is related object'] = df_hl_obj['related object'].apply(lambda x: obj in x)
    #    df_hl_obj = df_hl_obj[df_hl_obj['is related object']==True]
    #    df_hl_obj.drop(columns='is related object')
    #    #print(obj)
    #    #print(len(df_hl_obj))
    #    if len(df_hl_obj['frequency'])>0:
    #        max_obj = int(max(df_hl_obj['frequency']))
    #    df_hl_obj['frequency'] = df_hl_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))
    #    filtered = df_hl_obj[df_hl_obj['frequency']==True]
    #    #filtered = df_hl_obj[df_hl_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))]
    #    filtered_df_hl = pd.concat([filtered_df_hl, filtered])
    filtered_df_hl = df_hl.copy()
    print('high load events after filtering:')
    print(len(filtered_df_hl))
    #print(filtered_df_hl)
        
    #filter high demand events by just keeping the highest demands per object
    df_hd = df.copy()
    df_hd = df_hd[df_hd['event type']=='hd']
    print(len(df_hd))
    df_hd.drop_duplicates(subset=['event', 'time segment'], inplace=True, keep='first')
    print('all high demand events:')
    print(len(df_hd))
    #df_hd = df_hd.astype({'frequency':'int'})
    #filtered_df_hd = pd.DataFrame()
    #all_inst = []
    #for index, row in filtered_df_hd.iterrows():
    #    curr_inst = row['related object']
    #    for i in curr_inst:
    #        all_inst.append(i)
    #all_inst_set = set(all_inst)
    #for obj in all_inst_set:
    #    df_hd_obj = df_hd.copy()
    #    df_hd_obj['is related object'] = df_hd_obj['related object'].apply(lambda x: obj in x)
    #    df_hd_obj = df_hd_obj[df_hd_obj['is related object']==True]
    #    df_hd_obj.drop(columns='is related object')
    #    #print(obj)
    #    #print(len(df_hl_obj))
    #    if len(df_hd_obj['frequency'])>0:
    #        max_obj = int(max(df_hd_obj['frequency']))
    #    df_hd_obj['frequency'] = df_hd_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))
    #    filtered = df_hd_obj[df_hd_obj['frequency']==True]
    #    #filtered = df_hl_obj[df_hl_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))]
    #    filtered_df_hd = pd.concat([filtered_df_hd, filtered])
    #filtered_df_hd = df_hd.sort_values('frequency', ascending=False)
    #num_hd = len(df_hd)
    #filtered_df_hd = filtered_df_hd.head(n=int(num_hd/2))
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
    #df_llt.sort_values('frequency', ascending=True, inplace=True)
    #number_lt = len(df_llt)
    #filtered_llt = df_llt.head(n=int(number_lt/10))
    filtered_llt = df_llt.copy()
    filtered_llt['event type'] = 'llt'
    print('low lagging time events after filter:')
    print(len(filtered_llt))

    #filter lagging time events by keeping those above 0.7*max lagging time
    df_lt = df.copy()
    df_lt = df_lt[df_lt['event type']=='lt']
    #df_lt.sort_values('frequency', ascending=False, inplace=True)
    #number_lt = len(df_lt)
    #filtered_lt = df_lt.head(n=int(number_lt/10))
    filtered_lt = df_lt.copy()
    filtered_lt['event type'] = 'hlt'
    print('high lagging time events after filter:')
    print(len(filtered_lt))

    #new_df_case = pd.concat([df_l, df_j, filtered_llt, filtered_lt], ignore_index=True)
    #new_df_obj = pd.concat([filtered_df_hl, filtered_df_hd, filtered_llt, filtered_lt], ignore_index=True)#df_s, df_m, df_l, df_j,
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
        if True:
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
                        if first_type == 'hd' or first_type == 'hl':
                            print('found hd/hl relation')
                        first_id = row['id']
                        second_id = r['id']
                        succ_list = result_dict[first_id]
                        succ_list.append(second_id)
                        result_dict[first_id] = succ_list
                        if first_type == 'm':
                            tw_df2 = new_df2.copy()
                            tw_df2 = tw_df2[tw_df2['time segment'] == time1+2]
                            for ind, ro in tw_df2.iterrows():
                                objects3 = ro['related object']
                                shared_objects2 = [value for value in objects1 if value in objects3] 
                                if not(shared_objects2 == []):
                                    #print('shared objects exist.')
                                    #print(shared_objects)
                                    type3 = ro['event type']
                                    third_type = ['hd', 'hl']
                                    if (type3 in third_type):
                                        third_id = ro['id']
                                        succ_list = result_dict[third_id]
                                        succ_list.append(first_id)
                                        result_dict[third_id] = succ_list
                case1 = row['quasi case id']
                case2 = r['quasi case id']
                if case1 == case2:
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
    print('node dict size:' + str(len(node_dict)))
    #print(result_dict)
    #print(cascade_dict)

    with open('./dicts/result_dict.pkl', 'wb') as fp:
        pickle.dump(result_dict, fp)
    print('result dictionary saved successfully to file')
    with open('./dicts/cascade_dict.pkl', 'wb') as fp:
        pickle.dump(cascade_dict, fp)
    print('cascade dictionary saved successfully to file')
    with open('./dicts/node_dict.pkl', 'wb') as fp:
        pickle.dump(node_dict, fp)
    print('node dictionary saved successfully to file')

    #G1 = nx.DiGraph(result_dict)
    #nx.draw(G1)
    #plt.savefig('G1')
    G2 = nx.Graph(node_dict)
    print('constructed graph')
    con_comp = nx.connected_components(G2)
    print('constructed components')
    S = [G2.subgraph(c).copy() for c in con_comp]
    print('constructed subgraphs')
    for i in range(10):
        nx.draw(S[i])
        plt.savefig('./graphs/S' + str(i))
    print('saved subgraphs')
    largest_cc = max(con_comp, key=len)
    nx.draw(largest_cc)
    plt.savefig('largest_cc')
    print('saved lagest cc')
    #nx.draw(G2)
    #plt.savefig('G2')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    #G3  = nx.DiGraph(cascade_dict)
    #nx.draw(G3)
    #plt.savefig('G3')
    nx.write_edgelist(G2, './graphs/G2.edgelist')
    #nx.write_edgelist(G1, './graphs/G1.edgelist')
    #nx.write_edgelist(G3, './graphs/G3.edgelist')
    #G = nx.DiGraph()

def construct_cascade():
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
        str_name = parent_path + '/hl_logs_new_new/hl_log' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)
        str_name = parent_path + '/hl_logs_new_new/hl_correction' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)

    df['id'] = df.index
    print('number of events:')
    print(len(df))
    print(df['time segment'].unique())

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
    df_hd = df.copy()
    df_hd = df_hd[df_hd['event type']=='hd']
    print(len(df_hd))
    df_hd.drop_duplicates(subset=['event', 'time segment'], inplace=True, keep='first')
    print('all high demand events:')
    print(len(df_hd))
    df_hd = df_hd.astype({'frequency':'int'})
    filtered_df_hd = pd.DataFrame()
    all_inst = []
    for index, row in df_hd.iterrows():
        curr_inst = row['related object']
        for i in curr_inst:
            all_inst.append(i)
    print(all_inst)
    all_inst_set = set(all_inst)
    print(all_inst_set)
    for obj in all_inst_set:
        df_hd_obj = df_hd.copy()
        df_hd_obj['is related object'] = df_hd_obj['related object']
        df_hd_obj['is related object'] = df_hd_obj['is related object'].apply(lambda x: obj in x)
        df_hd_obj = df_hd_obj[df_hd_obj['is related object']==True]
        df_hd_obj.drop(columns='is related object')
        #print(obj)
        #print(len(df_hl_obj))
        if len(df_hd_obj['frequency'])>0:
            max_obj = int(max(df_hd_obj['frequency']))
        df_hd_obj['frequency'] = df_hd_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))
        filtered = df_hd_obj[df_hd_obj['frequency']==True]
        #filtered = df_hl_obj[df_hl_obj['frequency'].apply(lambda x: int(x) > (max_obj * 0.5))]
        filtered_df_hd = pd.concat([filtered_df_hd, filtered])
    #filtered_df_hd = df_hd.sort_values('frequency', ascending=False)
    #num_hd = len(df_hd)
    #filtered_df_hd = filtered_df_hd.head(n=int(num_hd/2))
    #filtered_df_hd = df_hd.copy()
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

    new_df = pd.concat([filtered_df_hl, filtered_df_hd, df_s, df_m, df_l, df_j, filtered_llt, filtered_lt], ignore_index=True)
    print('number of high level events after filering:')
    print(len(new_df))

    with open('node_dict.pkl', 'rb') as fp:
        node_dict = pickle.load(fp)

    G2 = nx.Graph(node_dict)
    print('constructed graph')
    con_comp = nx.connected_components(G2)
    print('constructed components')
    S = [G2.subgraph(c).copy() for c in con_comp]
    print('number of cc: ' + str(len(S)))
    cascade_string = ''
    count = 0
    for sg in S:
        all_firsts = set()
        all_seconds = set()
        count = count +1
        print(count)
        nodes = list(sg.nodes())
        for node in nodes:
            tmp_df = new_df.copy()
            tmp_df = tmp_df[tmp_df['id']==node]
            if len(tmp_df)>1:
                print('problem')
            for index,row in tmp_df.iterrows():
                et1 = row['event type']
            successors = node_dict.get(node, [])
            for succ in successors:
                all_firsts.add(node)
                all_seconds.add(succ)
                tmp_df = new_df.copy()
                tmp_df = tmp_df[tmp_df['id']==succ]
                if len(tmp_df)>1:
                    print('problem')
                for index,row in tmp_df.iterrows():
                    et2 = row['event type']
                et_list = ['hd', 'hl']
                if ((et1 in et_list)):# or (et2 in et_list)):
                    print('take' + str(count))
                cascade_string = cascade_string + ', (' + str(node) + '): ' + str(et1) + ' -> (' + str(succ) + '): ' + str(et2)
        three_way = all_firsts.intersection(all_seconds)
        if three_way != set():
            print('take' + str(count))
        save_name = 'cascade' + str(count) + '.txt'
        file_obj = open(save_name, "w")
        file_obj.write(cascade_string)
        file_obj.close()
        cascade_string = ''

#main()
pattern_detection_dict2()
#case_construction()
#construct_cascade()



#add high level event of low lagging time per activity to log
#                for i in early_index:
#                    fastest_obj_type = predecessor.loc[i, 'ocel:type']
#                    fastest_object = predecessor.loc[i, 'object']
#                    if object_handling[lagging_obj_type]:
#                        #object type
#                        event.append('low lagging time of object type ' + str(fastest_obj_type) + ' in activity ' + str(row['ocel:activity']))
#                        event_type.append('llt')
#                        frequency.append(delay_time)
#                        start_time.append(min_time + (curr_seg_count-1)*time_segment)
#                        end_time.append(min_time + curr_seg_count*time_segment)
#                        time_frame.append(curr_seg_count)
#                        qcid.append(case)
#                        related_object.append(fastest_obj_type)
#                    else:
#                        #object instance
#                        event.append('low lagging time of object ' + str(fastest_object) + ' in activity ' + str(row['ocel:activity']))
#                        event_type.append('llt')
#                        frequency.append(delay_time)
#                        start_time.append(min_time + (curr_seg_count-1)*time_segment)
#                        end_time.append(min_time + curr_seg_count*time_segment)
#                        time_frame.append(curr_seg_count)
#                        qcid.append(case)
#                        related_object.append(fastest_object)