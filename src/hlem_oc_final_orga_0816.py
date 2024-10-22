import pm4py
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import networkx as nx
import pickle
import ast
from datetime import datetime
pd.options.mode.chained_assignment = None  # default='warn'


def main():
    #define data set and paths
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    dataset = 'order-management2'
    #dataset = 'ocel2-p2p'
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
    #number_segments = 943 #roughly days for ocel2-p2p dataset
    time_bounds = get_timebounds(min_time, max_time, number_segments)
    #create mapping of events on time windows
    time_of_event = dict()

    #partition df according to time bounds
    collect_log = pd.DataFrame()
    for j in range(0,number_segments,1):
        #print('execution: ' + str(j))
        partitioned_df = df.copy()
        if j<number_segments-1:
            partitioned_df = partitioned_df[partitioned_df['ocel:timestamp']<=time_bounds[j]]
        if j>0:
            partitioned_df = partitioned_df[partitioned_df['ocel:timestamp']>time_bounds[j-1]]
        #fill time dict
        for index,row in partitioned_df.iterrows():
            time_of_event[row['ocel:eid']] = j
        #high load and high demand
        hl_features_hl_hd = compute_hl_hd(partitioned_df, object_types, object_handling, j)
        collect_log = pd.concat([collect_log, hl_features_hl_hd], ignore_index=True)
    
    print(collect_log)
    #split/joins and syn time
    hl_feature_split_join = compute_split_join_syn(df, oc_rel_df, time_of_event, object_types)
    collect_log = pd.concat([collect_log, hl_feature_split_join], ignore_index=True)
    print(collect_log)
    #enter/exit segment
    hl_feature_enter_exit = compute_enter_exit(time_of_event, oc_rel_df)
    collect_log = pd.concat([collect_log, hl_feature_enter_exit], ignore_index=True)
    print(collect_log)
    #add high-level event id
    hl_eid = list(range(len(collect_log)))
    collect_log['hl:eid'] = hl_eid
    print(collect_log)
    #save high-level log
    save_name = parent_path + '/hl_log_orga_0822_ocel2-p2p.csv'
    collect_log.to_csv(save_name)

    hl_event_types = ['enter', 'exit', 's', 'j', 'syn', 'hl', 'hd']
    #hl_log = compute_hl_events(collect_log, hl_event_types, thresholds)
    pattern_df = collect_log.copy()
    thresholds = compute_thresholds(pattern_df, hl_event_types)
    print(thresholds)
    #adapt timestamps in df to make threshold work
    for index,row in pattern_df.iterrows():
        if row['feature type']=='syn':
            if type(row['feature value']==type('string')):
                print('time is string')
                new_time = convert_string_time(row['feature value'])
                pattern_df.loc[index, 'feature value'] = new_time
    print(pattern_df['feature value'])
    hl_log = compute_hl_events(pattern_df, hl_event_types, thresholds)
    print(len(hl_log))
    save_name = parent_path + '/filtered_hl_events_ocel2-p2p.csv'
    hl_log.to_csv(save_name)
    hl_relations = compute_relations(hl_log, number_segments, objects)

def start_with_log_csv():
    #define data set and path
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    dataset = 'order-management2'
    #dataset = 'ocel2-p2p'
    path = parent_path + './input_data/' + dataset + '.sqlite'
    log, df = read_input(path)
    obj_sum = get_obj_sum(log)
    objects = get_object_instances(obj_sum)
    num_time_windows = 428
    #num_time_windows = 563
    print(objects)
    print('number ob objects:')
    print(len(objects))

    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '/hl_log_orga_0822.csv'
    #str_name = parent_path + '/hl_log_orga_0822_ocel2-p2p.csv'
    pattern_df = pd.read_csv(str_name)
    print(pattern_df)
    hl_event_types = ['enter', 'exit', 's', 'j', 'syn', 'hl', 'hd']
    #by saving the list of objects, it gets a string. therefore the conversion is necessary
    print(pattern_df)
    pattern_df['related object'] = convert_string_to_list(pattern_df['related object'])
    print(pattern_df)
    pattern_df, thresholds = compute_thresholds(pattern_df, hl_event_types, num_time_windows)
    print(thresholds)
    #adapt timestamps in df to make threshold work
    for index,row in pattern_df.iterrows():
        if row['feature type']=='syn':
            new_time = convert_string_time(row['feature value'])
            pattern_df.loc[index, 'feature value'] = new_time
    print(pattern_df['feature value'])
    hl_log = compute_hl_events(pattern_df, hl_event_types, thresholds)
    print(len(hl_log))
    save_name = parent_path + '/filtered_hl_events.csv'
    #save_name = parent_path + '/filtered_hl_events_ocel2-p2p.csv'
    hl_log.to_csv(save_name)
    hl_relations = compute_relations(hl_log, num_time_windows, objects)

def compute_relations(input_df, input_number_win, input_objects):
    #compute time candidates:
    #set time window threshold
    num_time_windows = input_number_win
    max_time_diff = 3
    time_event_candidates_firsts, time_event_candidates_seconds = compute_time_based_relations2(input_df, max_time_diff, num_time_windows) 
    #create time relation dict
    time_candidates_dict = dict()
    for eid in set(time_event_candidates_firsts):
        time_candidates_dict[eid] = []
    for i in range(len(time_event_candidates_firsts)):
        eid1 = time_event_candidates_firsts[i]
        eid2 = time_event_candidates_seconds[i]
        curr_succ = time_candidates_dict[eid1]
        curr_succ.append(eid2)
        time_candidates_dict[eid1] = curr_succ
    print('test if dict construction works:')
    object_event_candidates_firsts, object_event_candidates_seconds = compute_object_based_relations2(input_df, input_objects)
    object_candidates_dict = dict()
    for eid in set(object_event_candidates_firsts):
        object_candidates_dict[eid] = []
    for i in range(len(object_event_candidates_firsts)):
        eid1 = object_event_candidates_firsts[i]
        eid2 = object_event_candidates_seconds[i]
        curr_succ = object_candidates_dict[eid1]
        curr_succ.append(eid2)
        object_candidates_dict[eid1] = curr_succ
    print('object candidates constructed')
    final_relations_dict = dict()
    for eid in time_event_candidates_firsts:
        time_succ = time_candidates_dict[eid]
        obj_succ = object_candidates_dict[eid]
        checked_succ = set(time_succ).intersection(set(obj_succ))
        final_relations_dict[eid] = checked_succ
    print('final dict constructed')
    all_first_eids = list(final_relations_dict.keys())
    first_eids = []
    second_eids = []
    for eid in all_first_eids:
        succ = final_relations_dict[eid]
        for s in succ:
            first_eids.append(eid)
            second_eids.append(s)
    print('final lists constructed')

    save_data = {'eid1': first_eids, 'eid2': second_eids}
    save_df = pd.DataFrame(data=save_data)
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    save_name = parent_path + '/final_relations.csv'
    #save_name = parent_path + '/final_relations_ocel2-p2p.csv'
    save_df.to_csv(save_name)


def compute_time_based_relations2(input_df, max_time_diff, num_time_windows):
    working_df = input_df.copy()
    candidates_firsts = []
    candidates_seconds = []
    tw_eid_dict = dict()
    for i in range(num_time_windows):
        tw_eids = working_df[working_df['time window']==i]['hl:eid'].tolist()
        tw_eid_dict[i] = tw_eids
    for i in range(num_time_windows):
        tw_eids = tw_eid_dict[i]
        for eid in tw_eids:
            for j in range(i,i+max_time_diff,1):
                if j<num_time_windows:
                    j_eids = tw_eid_dict[j]
                    for e in j_eids:
                        candidates_firsts.append(eid)
                        candidates_seconds.append(e)
    return candidates_firsts, candidates_seconds

def list_containment_test(list, val):
    if val in list:
        return list
    else:
        return 'nan'

def compute_object_based_relations2(input_df, objects):
    working_df = input_df.copy()
    candidates_firsts = []
    candidates_seconds = []
    object_eid_dict = dict()
    for obj in objects:
        obj = obj.strip()
        object_df = working_df.copy()
        object_df['related object'] = object_df['related object'].apply(lambda x: list_containment_test(x, obj))
        final_object_df = object_df[object_df['related object'] != 'nan']
        object_eids = final_object_df['hl:eid'].tolist()
        object_eid_dict[obj] = object_eids
    print('dict created')
    for obj in objects:
        obj = obj.strip()
        for eid1 in object_eid_dict[obj]:
            for eid2 in object_eid_dict[obj]:
                candidates_firsts.append(eid1)
                candidates_seconds.append(eid2)
    print('lists constructed')
    print('len of obj candidates:')
    print(len(candidates_firsts))
    print(len(candidates_seconds))
    return candidates_firsts, candidates_seconds

def convert_string_to_list(input_coulmn):
    working_col = input_coulmn.copy()
    working_col = working_col.apply(lambda x: x.split(','))
    working_col = working_col.apply(lambda x: own_del(x))
    return working_col

def own_del(x):
    new_x = []
    for y in x:
        new_y = y
        new_y = new_y.strip()
        new_y = new_y.replace("[", '')
        new_y = new_y.replace("]", '')
        new_y = new_y.replace('\'', '')
        new_x.append(new_y)
    return new_x

def evaluate_threshold(input_df, input_threshold):
    output_df = input_df.copy()
    output_df = output_df[output_df['feature value']>= str(input_threshold)]
    #test wenn direkt in einem durchlaifen
    #output_df = output_df[output_df['feature value']>= input_threshold]
    return output_df

def compute_hl_events(input_df, input_types, input_thesholds):
    i = 0
    hl_events = pd.DataFrame()
    for t in input_types:
        t_df = input_df.copy()
        t_df = t_df[t_df['feature type']==t]
        t_events = evaluate_threshold(t_df, input_thesholds[i])
        hl_events = pd.concat([hl_events, t_events])
        i = i + 1
    return hl_events

def compute_enter_exit_threshold(input_df):
    working_df = input_df.copy()
    working_df = working_df.drop(working_df[working_df['feature value'] == '0'].index)
    values = working_df['feature value'].tolist()
    int_values = []
    for v in values:
        int_values.append(int(v))
    sum_of_values = sum(int_values)
    threshold = sum_of_values/len(values)
    return threshold

def compute_split_join_threshold(input_df):
    values = input_df['feature value'].tolist()
    int_values = []
    for v in values:
        int_values.append(int(v))
    threshold = min(int_values)
    return threshold

def compute_syn_threshold(input_df):
    values = input_df['feature value'].tolist()
    #in the conversion date changes to string, this messes the sort up, therefore the string conversion is necessary here
    converted_values = convert_string_times(values)
    #sorted_values = sorted(values)
    sorted_values = sorted(converted_values)
    threshold = sorted_values[int(len(values)/2)]
    return threshold

def convert_string_times(input_values):
    converted_values = []
    for v in input_values:
        if len(v)==15:
            new_v = '0' + v
            converted_values.append(new_v)
        else:
            converted_values.append(v)
    return converted_values

def convert_string_time(input_value):
    if len(input_value)==15:
        new_v = '0' + input_value
        return new_v
    else:
        return input_value

def compute_hl_hd_threshold(input_df):
    input_df = input_df.drop(input_df[input_df['feature value'] == '0'].index)
    if len(input_df)>0:
        values = input_df['feature value'].tolist()
        int_values = []
        for v in values:
            int_values.append(int(v))
        sorted_values = sorted(int_values)
        threshold = sorted_values[int(len(values)/2)]
        return threshold
    else:
        return 0
    
def filter_spread(input_df, input_threshold):
    working_df = input_df.copy()
    working_df['feature value'] = working_df['feature value'].apply(lambda x: x if int(x)>input_threshold else np.nan)
    working_df.dropna(subset=['feature value'],inplace=True)
    return working_df

def filter_split_join_events(input_df, num_tw):
    spread_threshold = 2
    input_df = filter_spread(input_df, spread_threshold)
    act1_dict, act2_dict = create_act_dict(input_df)
    duplicated_dict = dict()
    unnecessary_dups = set()
    for i in range(num_tw):
        i_df = input_df.copy()
        i_df = i_df[i_df['time window']==i]
        i_ids = set(i_df['hl:eid'].tolist())
        for id1 in i_ids:
            duplicated_dict[id1] = []
            for id2 in i_ids:
                if id2 > id1:
                    act11 = act1_dict[id1]
                    act21 = act2_dict[id1]
                    act12 = act1_dict[id2]
                    act22 = act2_dict[id2]
                    if (set(act11) == set(act12)) and (set(act21) == set(act22)):
                        found_dups = duplicated_dict[id1]
                        found_dups.append(id2)
                        duplicated_dict[id1] = found_dups
                        unnecessary_dups.add(id2)
    feature_type = []
    feature_value = []
    rel_obj = []
    rel_event = []
    seg_act1 = []
    seg_act2 = []
    timestamp = []
    time_window = []
    hl_eid = []
    for index, row in input_df.iterrows():
        id = row['hl:eid']
        if id not in unnecessary_dups:
            dups = duplicated_dict[id]
            if dups == []:
                feature_type.append(row['feature type'])
                feature_value.append(row['feature value'])
                rel_obj.append(row['related object'])
                rel_event.append(row['related event'])
                seg_act1.append(row['seg act1'])
                seg_act2.append(row['seg act2'])
                timestamp.append(row['timestamp'])
                time_window.append(row['time window'])
                hl_eid.append(row['hl:eid'])
            else:
                new_feature = row['feature type']
                new_value = int(row['feature value'])
                new_obj = row['related object']
                new_event = row['related event']
                new_act1 = row['seg act1']
                new_act2 = row['seg act2']
                new_timestamp = row['timestamp']
                new_tw = row['time window']
                new_eid = row['hl:eid']
                for dup in dups:
                    id2 = input_df.index[input_df['hl:eid'] == dup].tolist()[0]
                    new_value = int(new_value) + int(1)
                    old_obj = input_df.loc[id2, 'related object']
                    for o in old_obj:
                        new_obj.append(o)
                feature_type.append(new_feature)
                feature_value.append(str(new_value))
                rel_obj.append(new_obj)
                rel_event.append(new_event)
                seg_act1.append(new_act1)
                seg_act2.append(new_act2)
                timestamp.append(new_timestamp)
                time_window.append(new_tw)
                hl_eid.append(new_eid)
    return_df = pd.DataFrame({'feature type': feature_type, 'feature value': feature_value, 'related object': rel_obj, 'related event': rel_event, 'seg act1': seg_act1, 'seg act2': seg_act2, 'timestamp': timestamp, 'time window': time_window, 'hl:eid': hl_eid})
    return return_df

def create_act_dict(input_df):
    act1_dict = dict()
    act2_dict = dict()
    for index, row in input_df.iterrows():
        eid = row['hl:eid']
        act1 = row['seg act1']
        act2 = row['seg act2']
        act1_dict[eid] = act1
        act2_dict[eid] = act2
    return act1_dict, act2_dict

def compute_thresholds(input_df, input_types, num_tw):
    new_df = pd.DataFrame()
    for t in input_types:
        if t=='enter':
            enter_df = input_df.copy()
            enter_df = enter_df[enter_df['feature type']=='enter']
            new_df =pd.concat([new_df,enter_df])
            enter_threshold = compute_enter_exit_threshold(enter_df)
        if t=='exit':
            exit_df = input_df.copy()
            exit_df = exit_df[exit_df['feature type']=='exit']
            new_df =pd.concat([new_df,exit_df])
            exit_threshold = compute_enter_exit_threshold(exit_df)
        if t=='s':
            split_df = input_df.copy()
            split_df = split_df[split_df['feature type']=='s']
            split_df = filter_split_join_events(split_df, num_tw)
            new_df =pd.concat([new_df,split_df])
            split_threshold = compute_split_join_threshold(split_df)
        if t=='j':
            join_df = input_df.copy()
            join_df = join_df[join_df['feature type']=='j']
            join_df = filter_split_join_events(join_df, num_tw)
            new_df =pd.concat([new_df,join_df])
            join_threshold = compute_split_join_threshold(join_df)
        if t=='syn':
            syn_df = input_df.copy()
            syn_df = syn_df[syn_df['feature type']=='syn']
            new_df =pd.concat([new_df,syn_df])
            syn_threshold = compute_syn_threshold(syn_df)
        if t=='hl':
            hl_df = input_df.copy()
            hl_df = hl_df[hl_df['feature type']=='hl']
            new_df =pd.concat([new_df,hl_df])
            hl_threshold = compute_hl_hd_threshold(hl_df)
        if t=='hd':
            hd_df = input_df.copy()
            hd_df = hd_df[hd_df['feature type']=='hd']
            new_df =pd.concat([new_df,hd_df])
            hd_threshold = compute_hl_hd_threshold(hd_df)
    thresholds = new_df, [enter_threshold, exit_threshold, split_threshold, join_threshold, syn_threshold, hl_threshold, hd_threshold]
    return thresholds

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
    for index, row in all_segments.iterrows():
        act1 = row['act1']
        act2 = row['act2']
        segment_df = input_rel.copy()
        segment_df = segment_df[segment_df['act1']==act1]
        segment_df = segment_df[segment_df['act2']==act2]
        time1_windows = []
        time2_windows = []
        for i, r in segment_df.iterrows():
            time1_windows.append(input_time[r['eid1']])
            time2_windows.append(input_time[r['eid2']])
        segment_df['time window 1'] = time1_windows
        segment_df['time window 2'] = time2_windows
        for tw in set(time1_windows):
            enter_df = segment_df[segment_df['time window 1'] == tw]
            passing_objects_enter = enter_df['object'].tolist()
            exit_df = segment_df[segment_df['time window 2'] == tw]
            passing_objects_exit = exit_df['object'].tolist()
            #enter event
            feature_type.append('enter')
            feature_value.append(len(enter_df))
            related_object.append(passing_objects_enter)
            related_event.append('nan')
            seg_act1.append(act1)
            seg_act2.append(act2)
            timestamp.append('nan')
            time_win.append(tw)
            #exit event
            feature_type.append('exit')
            feature_value.append(len(exit_df))
            related_object.append(passing_objects_exit)
            related_event.append('nan')
            seg_act1.append(act1)
            seg_act2.append(act2)
            timestamp.append('nan')
            time_win.append(tw)
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

def get_object_instances_of_event(input_row, input_types):
    output_objects = []
    for ot in input_types:
        str_name = 'ocel:type:' + str(ot)
        for oi in input_row[str_name]:
            output_objects.append(oi)
    return output_objects

def compute_split_join_syn(input_df, input_rel, input_times, input_types):
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
            #get all objects
            objects_of_event = get_object_instances_of_event(row, input_types)
            f_value = len(succ_df)
            #create high level feature of split
            feature_type.append('s')
            #value of split is not necessaryly correct here. Spread size recognized in relation computation
            feature_value.append(f_value)
            related_object.append(objects_of_event)
            related_event.append(eid)
            seg_act1.append(row['ocel:activity'])
            seg_act2.append(succ_df['act2'].unique())
            timestamp.append(f_time)
            time_win.append(f_window)
        pred_df = input_rel.copy()
        pred_df.drop_duplicates(subset=['eid1', 'eid2'], inplace=True)
        pred_df = pred_df[pred_df['eid2']==eid]
        if len(pred_df)>1:
            #get all objects
            objects_of_event = get_object_instances_of_event(row, input_types)
            f_value = len(pred_df)
            syn_result = compute_syn_time(pred_df)
            syn_time = syn_result[0] 
            syn_eid = syn_result[1]
            syn_obj = syn_result[2]
            #create high level feature of join
            feature_type.append('j')
            #value of join is not necessaryly correct here. Spread size recognized in relation computation
            feature_value.append(f_value)
            related_object.append(objects_of_event)#syn_obj is the objects the process is waiting for
            related_event.append(eid)
            seg_act1.append(pred_df['act1'].unique())
            seg_act2.append(row['ocel:activity'])
            timestamp.append(f_time)
            time_win.append(f_window)
            #create high level feature of high synchronization time
            feature_type.append('syn')
            #value is not correct according to formal definition
            #impl: compute all events of any synchronization time
            #postprocessing needed: find t_syn threshold, filter hl events, comput value: number of hl events per timewindow
            feature_value.append(syn_time)
            related_object.append(objects_of_event)
            related_event.append(eid)
            seg_act1.append(pred_df['act1'].unique())
            seg_act2.append(row['ocel:activity'])
            timestamp.append(f_time)
            time_win.append(f_window)
    return_df = pd.DataFrame({'feature type': feature_type, 'feature value': feature_value, 'related object': related_object, 'related event': related_event, 'seg act1': seg_act1, 'seg act2': seg_act2, 'timestamp': timestamp, 'time window': time_win})
    return return_df

def get_interaction_of_object(input_df, input_type, input_instance, input_all_types):
    str_name = 'ocel:type:' + str(input_type)
    output_inst = []
    for index,row in input_df.iterrows():
        if input_instance in row[str_name]:
            event_inst = get_object_instances_of_event(row, input_all_types)
            for i in event_inst:
                output_inst.append(i)
    return output_inst

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
            obj_type_appear_count = 0
            for index,row in obj_df.iterrows():
                if row[str_name] != []:
                    obj_type_appear_count = obj_type_appear_count + 1
                    for o in row[str_name]:
                        obj_list.append(o)
            if input_handling[obj]:#type matters
                #high demand
                demand = len(set(obj_list))
                #create high level feature of high demand
                feature_type.append('hd')
                feature_value.append(demand)
                related_object.append(obj_list)#obj is the object type which has high demand
                related_event.append('nan')#evtl noch anpassen
                seg_act1.append('nan')
                seg_act2.append('nan')
                timestamp.append('nan')
                time_win.append(input_window)
            else:
                #high load
                for o in set(obj_list):
                    interactive_objects = get_interaction_of_object(obj_df, obj, o, input_types)
                    load = obj_list.count(o)
                    #create high level feature of high load
                    feature_type.append('hl')
                    feature_value.append(load)
                    related_object.append(interactive_objects)#o is the object of high load
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
    if input_data == 'ocel2-p2p':
        object_handling['goods receipt'] = True
        object_handling['invoice receipt'] = True
        object_handling['material'] = True
        object_handling['payment'] = True
        object_handling['purchase_order'] = True
        object_handling['purchase_requisition'] = True  
        object_handling['quotation'] = True
    return object_handling 

def get_object_types(input_log):
    return pm4py.ocel_get_object_types(input_log)

def read_input(input_path):
    output_log = pm4py.read_ocel2_sqlite(input_path)
    output_df = output_log.get_extended_table()
    return output_log, output_df

def compute_feature_dict(input_df):
    feature_dict = dict()
    for index, row in input_df.iterrows():
        hl_id = row['hl:eid']
        feature_dict[hl_id] = row['feature type']
    return feature_dict

def get_feature_types(input_rel, input_events):
    working_rel = input_rel.copy()
    working_events = input_events.copy()
    feature_dict = compute_feature_dict(working_events)
    feature1 = []
    feature2 = []
    for index, row in working_rel.iterrows():
        feature1.append(feature_dict[row['eid1']])
        feature2.append(feature_dict[row['eid2']])
    working_rel['feature type1'] = feature1
    working_rel['feature type2'] = feature2
    return working_rel
    
def get_relation_summary(input_df, input_types):
    f = open("relation_summary_ocel2-p2p.txt", "w")
    for t in input_types:
        text = t + ': '
        print(text)
        f.write(text)
        t_df = input_df.copy()
        t_df = t_df[t_df['feature type1']==t]
        text = 'number of relation where ' + str(t) + ' appears first: ' + str(len(t_df))
        print(text)
        f.write(text)
        print('second type split into: ')
        f.write('second type split into: ')
        for t2 in input_types:
            t2_df = t_df.copy()
            t2_df = t2_df[t2_df['feature type2']==t2]
            text = str(t2) + ': ' + str(len(t2_df))
            print(text)
            f.write(text)
    f.close()

def start_with_relations():
    #read relations
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    #str_name = parent_path + '/hl_log_orga_0822.csv'
    str_name = parent_path + '/final_relations_ocel2-p2p.csv'
    relations = pd.read_csv(str_name)

    #read hl events
    #str_name = parent_path + '/filtered_hl_events.csv'
    str_name = parent_path + '/filtered_hl_events_ocel2-p2p.csv'
    hl_events = pd.read_csv(str_name)

    relation_df = get_feature_types(relations, hl_events)
    hl_event_types = ['enter', 'exit', 's', 'j', 'syn', 'hl', 'hd']
    get_relation_summary(relation_df, hl_event_types)


#main()
start_with_log_csv()
#start_with_relations()