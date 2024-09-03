import pm4py
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import networkx as nx
import pickle
import ast
pd.options.mode.chained_assignment = None  # default='warn'


def main():
    #define data set and path
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    dataset = 'order-management2'
    path = parent_path + './input_data/' + dataset + '.sqlite'
    print(path)
    #read log and dataframe
    log, df = read_input(path)
    #sort by timestamp
    df.sort_values(by='ocel:timestamp', inplace=True)
    print(df)
    #get all object types
    object_types = get_object_types(log)
    print(object_types)
    #replace nan by []
    df = convert_nan(df, object_types)
    #get object summary
    obj_sum = get_obj_sum(log)
    #get all objects
    objects = get_object_instances(obj_sum)
    #compute types of object instances, return dicts inst->type, type->inst
    type_of_inst, all_object_inst = compute_instance_type_dicts(df, object_types)
    #set handling of object type: True=only type matters, not single instance, False=instance matters
    object_handling = set_object_handling(dataset)
    print(object_handling)
    #compute object-centric successor and predecessor
    oc_rel_df = get_oc_pred_succ(df, objects, object_handling, type_of_inst) #the df contains all segments: relations between events appesar multiple times for multiple objects
    #define time bounds
    min_time, max_time = define_timebounds(df)
    number_segments = 428 #roughly days for order-management dataset
    time_bounds = get_timebounds(min_time, max_time, number_segments)
    #create mapping of events on time windows
    time_of_event = dict()

    #partition df according to time bounds
    collect_log = pd.DataFrame()
    for j in range(0,number_segments,1):
        #print('execution: ' + str(j))
        partitioned_df = df.copy()
        partitioned_df = partitioned_df[partitioned_df['ocel:timestamp']<time_bounds[j]]
        if j>0:
            partitioned_df = partitioned_df[partitioned_df['ocel:timestamp']>=time_bounds[j-1]]
        #print(partitioned_df)

        #fill time dict
        for index,row in df.iterrows():
            time_of_event[row['ocel:eid']] = j
        
        #high load and high demand
        hl_features_hl_hd = compute_hl_hd(partitioned_df, object_types, object_handling, j)
        collect_log = pd.concat([collect_log, hl_features_hl_hd], ignore_index=True)

    print(collect_log)
    #split/joins and syn time
    hl_feature_split_join = compute_split_join_syn(df, oc_rel_df, time_of_event)
    collect_log = pd.concat([collect_log, hl_feature_split_join], ignore_index=True)
    print(collect_log)
    #enter/exit segment
    hl_feature_enter_exit = compute_enter_exit(time_of_event, oc_rel_df)
    collect_log = pd.concat([collect_log, hl_feature_enter_exit], ignore_index=True)
    print(collect_log)
    hl_eid = list(range(len(collect_log)))
    collect_log['hl:eid'] = hl_eid
    print(collect_log)
    save_name = parent_path + '/hl_log_orga_0822.csv'
    collect_log.to_csv(save_name)

def compute_enter_exit(input_time, input_rel):
    feature_type = []
    feature_value = []
    related_object = []
    related_event = []
    seg_act1 = []
    seg_act2 = []
    timestamp = []
    time_win = []
    all_segments = input_rel.copy()
    all_segments.drop_duplicates(subset=['act1','act2'], inplace=True)
    print(all_segments)
    for index, row in all_segments.iterrows():
        act1 = row['act1']
        act2 = row['act2']
        segment_df = input_rel.copy()
        segment_df = segment_df[segment_df['act1']==act1]
        segment_df = segment_df[segment_df['act2']==act2]
        for i, r in segment_df.iterrows():
            #enter event
            time_window = input_time[r['eid1']]
            feature_type.append('enter')
            feature_value.append('nan')
            related_object.append(r['object'])
            related_event.append(r['eid1'])
            seg_act1.append(r['act1'])
            seg_act2.append(r['act2'])
            timestamp.append(r['time1'])
            time_win.append(time_window)
            #exit event
            time_window = input_time[r['eid2']]
            feature_type.append('exit')
            feature_value.append('nan')
            related_object.append(r['object'])
            related_event.append(r['eid2'])
            seg_act1.append(r['act1'])
            seg_act2.append(r['act2'])
            timestamp.append(r['time2'])
            time_win.append(time_window)
    return_df = pd.DataFrame({'feature type': feature_type, 'feature value': feature_value, 'related object': related_object, 'related event': related_event, 'seg act1': seg_act1, 'seg act2': seg_act2, 'timestamp': timestamp, 'time window': time_win})
    return return_df


def convert_nan(input_df, input_types):
    for t in input_types:
        str_name = 'ocel:type:' + str(t)
        input_df[str_name] = input_df[str_name].fillna(0)
        input_df[str_name] = input_df[str_name].apply(lambda x: list() if x==0 else x)
    return input_df


def compute_syn_time(input_df):
    output_time = max(input_df['time1'])-min(input_df['time1'])
    output_id = input_df.loc[input_df['time1'].idxmax(), 'eid1']
    output_obj = input_df.loc[input_df['time1'].idxmax(), 'object']
    return [output_time, output_id, output_obj]

def compute_split_join_syn(input_df, input_rel, input_times):
    feature_type = []
    feature_value = []
    related_object = []
    related_event = []
    seg_act1 = []
    seg_act2 = []
    timestamp = []
    time_win = []
    for index, row in input_df.iterrows():
        eid = row['ocel:eid']
        f_time = row['ocel:timestamp']
        f_window = input_times[eid]
        succ_df = input_rel.copy()
        succ_df.drop_duplicates(subset=['eid1', 'eid2'], inplace=True)
        succ_df = succ_df[succ_df['eid1']==eid]
        if len(succ_df)>1:
            f_value = len(succ_df)
            #create high level feature of split
            feature_type.append('s')
            feature_value.append(f_value)
            related_object.append('nan')#noch anpassen
            related_event.append(eid)
            seg_act1.append(row['ocel:activity'])
            seg_act2.append(succ_df['act2'].unique())
            timestamp.append(f_time)
            time_win.append(f_window)
        pred_df = input_rel.copy()
        pred_df.drop_duplicates(subset=['eid1', 'eid2'], inplace=True)
        pred_df = pred_df[pred_df['eid2']==eid]
        if len(pred_df)>1:
            f_value = len(pred_df)
            syn_result = compute_syn_time(pred_df)
            syn_time = syn_result[0] 
            syn_eid = syn_result[1]
            syn_obj = syn_result[2]
            #create high level feature of join
            feature_type.append('j')
            feature_value.append(f_value)
            related_object.append(syn_obj)
            related_event.append(eid)
            seg_act1.append(pred_df['act1'].unique())
            seg_act2.append(row['ocel:activity'])
            timestamp.append(f_time)
            time_win.append(f_window)
            #create high level feature of high synchronization time
            feature_type.append('syn')
            feature_value.append(syn_time)
            related_object.append(syn_eid)#noch anpassen
            related_event.append(eid)
            seg_act1.append(pred_df['act1'].unique())
            seg_act2.append(row['ocel:activity'])
            timestamp.append(f_time)
            time_win.append(f_window)
    return_df = pd.DataFrame({'feature type': feature_type, 'feature value': feature_value, 'related object': related_object, 'related event': related_event, 'seg act1': seg_act1, 'seg act2': seg_act2, 'timestamp': timestamp, 'time window': time_win})
    return return_df

def compute_hl_hd(input_df, input_types, input_handling, input_window):
    feature_type = []
    feature_value = []
    related_object = []
    related_event = []
    seg_act1 = []
    seg_act2 = []
    timestamp = []
    time_win = []
    for obj in input_types:
            str_name = 'ocel:type:' + str(obj)
            obj_df = input_df.copy()
            obj_list = []
            for index,row in obj_df.iterrows():
                for o in row[str_name]:
                    obj_list.append(o)
            if input_handling[obj]:#type matters
                #high demand
                demand = len(set(obj_list))
                #create high level feature of high demand
                feature_type.append('hd')
                feature_value.append(demand)
                related_object.append(obj)
                related_event.append('nan')#evtl noch anpassen
                seg_act1.append('nan')
                seg_act2.append('nan')
                timestamp.append('nan')
                time_win.append(input_window)
            else:
                #high load
                for o in set(obj_list):
                    load = obj_list.count(o)
                    #create high level feature of high load
                    feature_type.append('hl')
                    feature_value.append(load)
                    related_object.append(o)
                    related_event.append('nan')#evtl noch anpassen
                    seg_act1.append('nan')
                    seg_act2.append('nan')
                    timestamp.append('nan')
                    time_win.append(input_window)
    return_df = pd.DataFrame({'feature type': feature_type, 'feature value': feature_value, 'related object': related_object, 'related event': related_event, 'seg act1': seg_act1, 'seg act2': seg_act2, 'timestamp': timestamp, 'time window': time_win})
    return return_df

def get_timebounds(in_min, in_max, in_seg):
    time_window = in_max-in_min
    time_segment = time_window/in_seg
    time_bounds = []
    for i in range(in_seg):
        time_bounds.append(in_min + (i+1)*time_segment)
    return time_bounds

def define_timebounds(input_df):
    min_time = min(input_df['ocel:timestamp'])
    max_time = max(input_df['ocel:timestamp'])
    return min_time, max_time

def get_oc_relations(input_df, input_object, input_type):
    str_name = 'ocel:type:' + str(input_type)
    filtered_df = input_df.copy()
    filtered_df[str_name] = filtered_df[str_name].apply(lambda x: True if input_object in x else False)
    filtered_df = filtered_df[filtered_df[str_name]==True]
    eid1 = []
    eid2 = []
    act1 = []
    act2 = []
    time1 = []
    time2 = []
    obj = []
    for i in range(len(filtered_df)-1):   
        eid1.append(filtered_df.iloc[i]['ocel:eid'])
        eid2.append(filtered_df.iloc[i+1]['ocel:eid'])
        act1.append(filtered_df.iloc[i]['ocel:activity'])
        act2.append(filtered_df.iloc[i+1]['ocel:activity'])
        time1.append(filtered_df.iloc[i]['ocel:timestamp'])
        time2.append(filtered_df.iloc[i+1]['ocel:timestamp'])
        obj.append(input_object)
    output_df = pd.DataFrame({'eid1':eid1, 'eid2': eid2, 'act1': act1, 'act2': act2, 'time1': time1, 'time2': time2, 'object': obj})
    #print(output_df)
    return output_df

def get_oc_pred_succ(input_df, input_objects, input_handling, input_dict):
    oc_rel_df = pd.DataFrame(columns=['eid1', 'eid2', 'act1', 'act2', 'time1', 'time2','object'])
    for obj in input_objects:
        type_of_obj = get_type_of_inst(obj, input_dict)
        if input_handling[type_of_obj]:
            tmp_df = get_oc_relations(input_df, obj, type_of_obj)
            oc_rel_df = pd.concat([oc_rel_df, tmp_df])
    #oc_rel_df.drop_duplicates(inplace=True)
    return oc_rel_df

def get_type_of_inst(input_object, input_dict):
    output_type = input_dict[input_object]
    return output_type

def compute_instance_type_dicts(input_df, input_types):
    compute_df = input_df.copy()
    #for t in input_types:
        #str_name = 'ocel:type:' + str(t)
        #compute_df[str_name] = compute_df[str_name].fillna(0)
        #compute_df[str_name] = compute_df[str_name].apply(lambda x: list() if x==0 else x)
    all_object_inst = {} #maps set of object on type
    for t in input_types:
        all_t_obj = []
        for id,row in compute_df.iterrows():
            str_name = 'ocel:type:' + str(t)
            all_t_obj = all_t_obj + row[str_name]
        all_object_inst[t] = set(all_t_obj)
    type_of_inst = {} #maps type on object
    for key in all_object_inst:
        value = all_object_inst[key]
        for v in value:
            type_of_inst[v]=key
    return type_of_inst, all_object_inst

def get_object_instances(input_sum):
    return input_sum['ocel:oid'].unique()

def get_obj_sum(input_log):
    return pm4py.ocel_objects_summary(input_log)

def set_object_handling(input_data):
    object_handling = {}
    #if input_data == '.procure-to-pay':
    #    for obj in object_types:
    #        object_handling[obj] = True
    if input_data == 'order-management2':
        object_handling['orders'] = True
        object_handling['items'] = True
        object_handling['packages'] = True
        object_handling['products'] = False
        object_handling['employees'] = False
        object_handling['customers'] = False  
    return object_handling 

def get_object_types(input_log):
    return pm4py.ocel_get_object_types(input_log)

def read_input(input_path):
    output_log = pm4py.read_ocel2_sqlite(input_path)
    output_df = output_log.get_extended_table()
    return output_log, output_df

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
    for i in range(333):
        #str_name = parent_path + '/hl_logs_new_new/hl_log' + str(i) + '.csv'
        str_name = parent_path + '/hl_logs_order-managemant/hl_log' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)
        #str_name = parent_path + '/hl_logs_new_new/hl_correction' + str(i) + '.csv'
        #input = pd.read_csv(str_name)
        #df = pd.concat([df, input], ignore_index=True)

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
    for index, row in filtered_df_hd.iterrows():
        curr_inst = row['related object']
        for i in curr_inst:
            all_inst.append(i)
    all_inst_set = set(all_inst)
    for obj in all_inst_set:
        df_hd_obj = df_hd.copy()
        df_hd_obj['is related object'] = df_hd_obj['related object'].apply(lambda x: obj in x)
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

    with open('./dicts_order-management/result_dict.pkl', 'wb') as fp:
        pickle.dump(result_dict, fp)
    print('result dictionary saved successfully to file')
    with open('./dicts_order-management/cascade_dict.pkl', 'wb') as fp:
        pickle.dump(cascade_dict, fp)
    print('cascade dictionary saved successfully to file')
    with open('./dicts_order-management/node_dict.pkl', 'wb') as fp:
        pickle.dump(node_dict, fp)
    print('node dictionary saved successfully to file')

    #G1 = nx.DiGraph(result_dict)
    #nx.draw(G1)
    #plt.savefig('G1')
    G2 = nx.Graph(node_dict)
    print('constructed graph')
    con_comp = nx.connected_components(G2)
    print('constructed components')
    #S = [G2.subgraph(c).copy() for c in con_comp]
    #print('constructed subgraphs')
    #for i in range(10):
    #    nx.draw(S[i])
    #    plt.savefig('S' + str(i))
    #print('saved subgraphs')
    #largest_cc = max(con_comp, key=len)
    #nx.draw(largest_cc)
    #plt.savefig('largest_cc')
    #print('saved lagest cc')
    #nx.draw(G2)
    #plt.savefig('G2')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    #G3  = nx.DiGraph(cascade_dict)
    #nx.draw(G3)
    #plt.savefig('G3')
    #nx.write_edgelist(G2, './graphs/G2.edgelist')
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
    for i in range(333):
        #str_name = parent_path + '/hl_logs_new_new/hl_log' + str(i) + '.csv'
        str_name = parent_path + '/hl_logs_order-managemant/hl_log' + str(i) + '.csv'
        input = pd.read_csv(str_name)
        df = pd.concat([df, input], ignore_index=True)
        #str_name = parent_path + '/hl_logs_new_new/hl_correction' + str(i) + '.csv'
        #input = pd.read_csv(str_name)
        #df = pd.concat([df, input], ignore_index=True)

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
    for index, row in filtered_df_hd.iterrows():
        curr_inst = row['related object']
        for i in curr_inst:
            all_inst.append(i)
    all_inst_set = set(all_inst)
    for obj in all_inst_set:
        df_hd_obj = df_hd.copy()
        df_hd_obj['is related object'] = df_hd_obj['related object'].apply(lambda x: obj in x)
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

    with open('./dicts_order-management/node_dict.pkl', 'rb') as fp:
        node_dict = pickle.load(fp)

    G2 = nx.Graph(node_dict)
    print('constructed graph')
    all_nodes = G2.nodes()
    id_index_dict = dict()
    for index, row in new_df.iterrows():
        node_id = row['id']
        if node_id in all_nodes:
            id_index_dict[node_id] = index

    print('constructed id index dict')
    con_comp = nx.connected_components(G2)
    print('constructed components')
    S = [G2.subgraph(c).copy() for c in con_comp]
    print('number of cc: ' + str(len(S)))
    cascade_string = ''
    cascade_string2 = ''
    count = 0
    for sg in S:
        all_firsts = set()
        all_seconds = set()
        count = count +1
        #print(count)
        nodes = list(sg.nodes())
        for node in nodes:
            index_of_node = id_index_dict[node]
            et1 = new_df.at[index_of_node, 'event type']
            successors = node_dict.get(node, [])
            for succ in successors:
                all_firsts.add(node)
                all_seconds.add(succ)
                index_of_succ = id_index_dict[succ]
                et2 = new_df.at[index_of_succ, 'event type']
                et_list = ['hd', 'hl']
                if ((et1 in et_list) or (et2 in et_list)):
                    print('take' + str(count))
                cascade_string = cascade_string + ', (' + str(node) + '): ' + str(et1) + ' -> (' + str(succ) + '): ' + str(et2)
                cascade_string2 = cascade_string2  + str(et1) + '->' + str(et2) + ','
        three_way = all_firsts.intersection(all_seconds)
        if three_way != set():
            print('take' + str(count))
        save_name = './cascade_txts_order-management/cascade' + str(count) + '.txt'
        file_obj = open(save_name, "w")
        file_obj.write(cascade_string)
        file_obj.close()
        save_name = './cascade_txts2_order-management/cascade' + str(count) + '.txt'
        file_obj = open(save_name, "w")
        file_obj.write(cascade_string2)
        file_obj.close()
        cascade_string = ''
        cascade_string2 = ''

main()
#pattern_detection_dict2()
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