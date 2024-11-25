import pm4py
import pandas as pd
import numpy as np
from collections import Counter
import collections
import matplotlib.pyplot as plt
import os
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pickle
import ast
from datetime import datetime
import sys
from graphviz import Digraph
pd.options.mode.chained_assignment = None  # default='warn'

#predefined datasets, please add your own if you want
DATASET = 'order-management2'
#DATASET = 'ocel2-p2p'

#predefined number of time windows, please adjust according to your dataset
NUM_TIME_WINDOWS = 428
#NUM_TIME_WINDOWS = 943


def main():
    #define data set and paths
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    path = parent_path + './input_data/' + DATASET + '.sqlite'
    print('current dataset: ' + path)
    #read log and dataframe
    log, df = read_input(path)
    #sort by timestamp
    df.sort_values(by='ocel:timestamp', inplace=True)
    #get all object types
    object_types = get_object_types(log)
    print('object types of the dataset: ' + str(object_types))
    #replace nan by []
    df = convert_nan(df, object_types)
    #get object summary
    obj_sum = get_obj_sum(log)
    #get all objects
    objects = get_object_instances(obj_sum)
    #compute types of object instances, return dicts inst->type, type->inst
    type_of_inst, all_object_inst = compute_instance_type_dicts(df, object_types)
    #set handling of object type: True=only type matters, not single instance, False=instance matters
    object_handling = set_object_handling(DATASET)
    print('current object handling: ' + str(object_handling) + '\nadjust in set_object_handling if necessary.')
    #compute object-centric successor and predecessor
    oc_rel_df = get_oc_pred_succ(df, objects, object_handling, type_of_inst) #the df contains all segments: relations between events appesar multiple times for multiple objects
    all_seg = oc_rel_df.copy()
    all_seg.drop_duplicates(subset=['act1', 'act2'], inplace=True)
    #define time bounds
    min_time, max_time = define_timebounds(df)
    number_segments = NUM_TIME_WINDOWS 
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
    #enter/exit segment
    hl_feature_enter_exit = compute_enter_exit(time_of_event, oc_rel_df)
    collect_log = pd.concat([collect_log, hl_feature_enter_exit], ignore_index=True)
    #add high-level event id
    hl_eid = list(range(len(collect_log)))
    collect_log['hl:eid'] = hl_eid
    #save high-level log
    save_name = parent_path + '/hl_event_candidates/hl_log_' + DATASET + '.csv'
    #save_name = parent_path + '/hl_log_orga_0822_ocel2-p2p.csv'
    collect_log.to_csv(save_name)
    print('saved high-level candidates to: ' + save_name)

def start_with_log_csv():
    #define data set and path
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    path = parent_path + './input_data/' + DATASET + '.sqlite'
    log, df = read_input(path)
    obj_sum = get_obj_sum(log)
    objects = get_object_instances(obj_sum)
    num_time_windows = NUM_TIME_WINDOWS

    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '/hl_event_candidates/hl_log_' + DATASET + '.csv'
    pattern_df = pd.read_csv(str_name)
    hl_event_types = ['enter', 'exit', 's', 'j', 'syn', 'hl', 'hd']
    #by saving the list of objects, it gets a string. therefore the conversion is necessary
    pattern_df['related object'] = convert_string_to_list(pattern_df['related object'])
    pattern_df, thresholds = compute_thresholds(pattern_df, hl_event_types, num_time_windows)
    #adapt timestamps in df to make threshold work, adapt other values too
    pattern_df['feature value'] = pattern_df['feature value'].apply(lambda x: convert_value(x))
    hl_log = compute_hl_events(pattern_df, hl_event_types, thresholds)
    save_name = parent_path + '/hl_events_filtered/filtered_hl_events_' + DATASET + '.csv'
    hl_log.to_csv(save_name)
    print('saved high-level events to: ' + save_name)
    hl_relations = compute_relations(hl_log, num_time_windows, objects)

def start_with_filtered_log():
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '\hl_events_filtered/filtered_hl_events_' + DATASET + '.csv'
    filtered_log = pd.read_csv(str_name)
    num_time_windows = NUM_TIME_WINDOWS
    path = parent_path + './input_data/' + DATASET + '.sqlite'
    log, df = read_input(path)
    obj_sum = get_obj_sum(log)
    objects = get_object_instances(obj_sum)
    hl_relations = compute_relations(filtered_log, num_time_windows, objects)

def convert_value(x):
    if type(x)==int:
        return x
    if len(x)==1:
        output = '00' + x
    elif len(x)==2:
        output = '0' + x
    elif len(x)==15:
        output = '0' + x
    else:
        output = x
    return output

def compute_relations(input_df, input_number_win, input_objects):
    #compute time candidates:
    #set time window threshold
    num_time_windows = input_number_win
    max_time_diff = 1
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
    object_event_candidates_firsts, object_event_candidates_seconds = compute_object_based_relations2(input_df, input_objects)
    object_candidates_dict = dict()
    all_firsts = set(object_event_candidates_firsts).union(set(time_event_candidates_firsts))
    for eid in all_firsts:
        object_candidates_dict[eid] = []
    for i in range(len(object_event_candidates_firsts)):
        eid1 = object_event_candidates_firsts[i]
        eid2 = object_event_candidates_seconds[i]
        curr_succ = object_candidates_dict[eid1]
        curr_succ.append(eid2)
        object_candidates_dict[eid1] = curr_succ
    final_relations_dict = dict()
    for eid in time_event_candidates_firsts:
        time_succ = time_candidates_dict[eid]
        obj_succ = object_candidates_dict[eid]
        checked_succ = set(time_succ).intersection(set(obj_succ))
        final_relations_dict[eid] = checked_succ
    all_first_eids = list(final_relations_dict.keys())
    first_eids = []
    second_eids = []
    for eid in all_first_eids:
        succ = final_relations_dict[eid]
        for s in succ:
            first_eids.append(eid)
            second_eids.append(s)

    save_data = {'eid1': first_eids, 'eid2': second_eids}
    save_df = pd.DataFrame(data=save_data)
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    save_name = parent_path + '/final_relations/final_relations_' + DATASET + '.csv'
    save_df.to_csv(save_name)
    print('saved relations to: ' + str(save_name))

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
            for j in range(i+1,i+max_time_diff+1):
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
    object_list = []
    candidates_tuples = []
    object_eid_dict = dict()
    eid_object_dict = dict()
    for index,row in input_df.iterrows():
        e = row['hl:eid']
        o = row['related object']
        eid_object_dict[e] = o
    for obj in objects:
        obj = obj.strip()
        object_df = working_df.copy()
        object_df['related object'] = object_df['related object'].apply(lambda x: list_containment_test(x, obj))
        final_object_df = object_df[object_df['related object'] != 'nan']
        object_eids = final_object_df['hl:eid'].tolist()
        object_eid_dict[obj] = object_eids
    for obj in objects:
        obj = obj.strip()
        for eid1 in object_eid_dict[obj]:
            for eid2 in object_eid_dict[obj]:
                object_list.append(obj)
                candidate_tuple = (eid1, eid2)
                candidates_tuples.append(candidate_tuple)
    candidate_tuple_objects = dict()
    for i in range(len(candidates_tuples)):
        candidate_tuple_objects[candidates_tuples[i]] = []
    for i in range(len(candidates_tuples)):
        curr_obj = candidate_tuple_objects[candidates_tuples[i]]
        new_obj = object_list[i]
        curr_obj.append(new_obj)
        candidate_tuple_objects[candidates_tuples[i]] = curr_obj
    fractions = []
    fraction_dict = dict()
    for event_tuple in list(candidate_tuple_objects.keys()):
        relation_objects = candidate_tuple_objects[event_tuple]
        obj1 = eid_object_dict[event_tuple[0]]
        obj2 = eid_object_dict[event_tuple[1]]
        union_obj = set(obj1).union(set(obj2))
        fraction = len(relation_objects)/len(union_obj)
        fractions.append(fraction)
        fraction_dict[event_tuple] = fraction
    final_firsts = []
    final_seconds = []
    for event_tuple in list(fraction_dict.keys()):
        frac = fraction_dict[event_tuple]
        if frac>= 0.35:
            final_firsts.append(event_tuple[0])
            final_seconds.append(event_tuple[1])
    return final_firsts, final_seconds

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

def evaluate_threshold(input_df, input_threshold, input_type):
    output_df = input_df.copy()
    int_list = ['hl', 'hd', 's', 'j', 'enter', 'exit']
    if input_type in int_list:
        if len(str(input_threshold))==1:
            str_threshold = '00' + str(input_threshold)
        if len(str(input_threshold))==2:
            str_threshold = '0' + str(input_threshold)
        output_df = output_df[output_df['feature value']>= str_threshold]
    else:
        output_df = output_df[output_df['feature value']>= input_threshold]
    return output_df

def evaluate_syn_threshold(input_df, input_threshold):
    values = input_df['feature value'].tolist()
    converted_values = convert_string_times(values)
    output_df = input_df.copy()
    output_df['feature value'] = converted_values
    output_df = output_df[output_df['feature value']>= input_threshold]
    return(output_df)

def compute_hl_events(input_df, input_types, input_thesholds):
    i = 0
    hl_events = pd.DataFrame()
    for t in input_types:
        t_df = input_df.copy()
        t_df = t_df[t_df['feature type']==t]
        t_events = evaluate_threshold(t_df, input_thesholds[i], t)
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
    int_values.sort()
    threshold = int_values[int(len(int_values)*0.7)]
    return int(threshold)

def compute_split_join_threshold(input_df):
    values = input_df['feature value'].tolist()
    int_values = []
    for v in values:
        int_values.append(int(v))
    int_values.sort()
    threshold = int_values[int(len(int_values)/2)]
    return threshold

def compute_int_syn_threshold(input_df):
    values = input_df['feature value'].tolist()
    int_values = []
    for v in values:
        int_values.append(int(v))
    int_values.sort()
    threshold = int_values[int(len(int_values)/2)]
    if threshold==1:
        print(threshold)
        threshold = 2
    return threshold

def compute_syn_threshold(input_df):
    values = input_df['feature value'].tolist()
    #in the conversion date changes to string, this messes the sort up, therefore the string conversion is necessary here
    converted_values = convert_string_times(values)
    #sorted_values = sorted(values)
    sorted_values = sorted(converted_values)
    threshold = sorted_values[int(len(converted_values)*0.5)]
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
        threshold = sorted_values[int(len(values)*0.7)]
        return threshold
    else:
        return 0
    
def filter_spread(input_df, input_threshold):
    working_df = input_df.copy()
    working_df['feature value'] = working_df['feature value'].apply(lambda x: x if int(x)>input_threshold else np.nan)
    working_df.dropna(subset=['feature value'],inplace=True)
    return working_df

def filter_split_join_events(input_df, num_tw, feature_type):
    spread_threshold = 1
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
                    if feature_type == 's':
                        act1 = act1_dict[id1]
                        act2 = act1_dict[id2]
                    if feature_type == 'j':
                        act1 = act2_dict[id1]
                        act2 = act2_dict[id2]
                    if act1==act2:
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
                #new act messes up the segments for act1 or act2 (dependent from split or join) but they are anyway not used
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

def filter_syn_threshold(input_df):
    threshold = compute_syn_threshold(input_df)
    output_df = evaluate_syn_threshold(input_df, threshold)
    print(output_df)
    return output_df

def filter_syn_events(input_df, num_tw):
    input_df = filter_syn_threshold(input_df)
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
                    act1 = act2_dict[id1]
                    act2 = act2_dict[id2]
                    if act1==act2:
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
                feature_value.append(1)#row['feature value'])
                rel_obj.append(row['related object'])
                rel_event.append(row['related event'])
                seg_act1.append(row['seg act1'])
                seg_act2.append(row['seg act2'])
                timestamp.append(row['timestamp'])
                time_window.append(row['time window'])
                hl_eid.append(row['hl:eid'])
            else:
                new_feature = row['feature type']
                times = []
                times.append(row['feature value'])
                new_obj = row['related object']
                new_event = row['related event']
                #same as split/join
                new_act1 = row['seg act1']
                new_act2 = row['seg act2']
                new_timestamp = row['timestamp']
                new_tw = row['time window']
                new_eid = row['hl:eid']
                for dup in dups:
                    id2 = input_df.index[input_df['hl:eid'] == dup].tolist()[0]
                    times.append(input_df.loc[id2, 'feature value'])
                    old_obj = input_df.loc[id2, 'related object']
                    for o in old_obj:
                        new_obj.append(o)
                times = convert_string_times(times)
                times.sort()
                num_times = len(times)
                new_value = num_times#times[int(num_times/2)]
                feature_type.append(new_feature)
                feature_value.append(new_value)
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

def convert_string_int(input_value):
    if len(input_value)==1:
        output_value = '00' + input_value
        return output_value
    if len(input_value)==2:
        output_value = '0' + input_value
        return output_value
    return input_value

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
            print(len(split_df))
            split_df = filter_split_join_events(split_df, num_tw, t)
            print(len(split_df))
            new_df =pd.concat([new_df,split_df])
            split_threshold = compute_split_join_threshold(split_df)
        if t=='j':
            join_df = input_df.copy()
            join_df = join_df[join_df['feature type']=='j']
            join_df = filter_split_join_events(join_df, num_tw, t)
            new_df =pd.concat([new_df,join_df])
            join_threshold = compute_split_join_threshold(join_df)
        if t=='syn':
            syn_df = input_df.copy()
            syn_df = syn_df[syn_df['feature type']=='syn']
            syn_df = filter_syn_events(syn_df, num_tw)
            new_df =pd.concat([new_df,syn_df])
            syn_threshold = compute_int_syn_threshold(syn_df)
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
        combined_time_windows = set(time1_windows)
        for t in time2_windows:
            combined_time_windows.add(t)
        for tw in set(combined_time_windows):
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
                    #interactive_objects = get_interaction_of_object(obj_df, obj, o, input_types)
                    load = obj_list.count(o)
                    #create high level feature of high load
                    feature_type.append('hl')
                    feature_value.append(load)
                    related_object.append(o)#o is the object of high load, interactive_objects
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
    #sets the object handling of the objet types of the dataset, predefined for the two predefnied dataset
    #add definition for other datasets
    #handling = True: means the object type only counts the instances and the instance is not of special interest, demand high-level events are constrcuted and the obejct types are used for dircetly-following relations
    #handling = True: means the object type contains multiple distinguish instances which are important, load high-level events are constrcuted and the obejct types are not used for dircetly-following relations
    object_handling = {}
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
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    open_path = parent_path + '/relation_stat/relation_summary_' + DATASET + '.txt'
    f = open(open_path, "w")
    for t in input_types:
        text = t + ': '
        f.write(text)
        t_df = input_df.copy()
        t_df = t_df[t_df['feature type1']==t]
        text = 'number of relation where ' + str(t) + ' appears first: ' + str(len(t_df)) + '\n'
        f.write(text)
        f.write('second type split into: \n')
        for t2 in input_types:
            t2_df = t_df.copy()
            t2_df = t2_df[t2_df['feature type2']==t2]
            text = str(t2) + ': ' + str(len(t2_df)) + '\n'
            f.write(text)
    f.close()

def create_event_candidate_statistic():
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '\hl_event_candidates\hl_log_' + DATASET + '.csv'
    candidates = pd.read_csv(str_name)
    #enter candidates
    enter_candidates = candidates.copy()
    enter_candidates = enter_candidates[enter_candidates['feature type'] == 'enter']
    enter_candidates['feature value'] = enter_candidates['feature value'].apply(lambda x: np.nan if x == str(0) else x)
    enter_candidates.dropna(subset=['feature value'], inplace=True)
    num_enter = len(enter_candidates)
    open_path = parent_path + '\event_candidates_stat\event_candidates_summary_' + DATASET + '.txt'
    f = open(open_path, "w")
    text = 'Number of all enter event candidates: ' + str(num_enter) + '\n'
    f.write(text)
    text = 'candidates per time window:\n'
    f.write(text)
    file_name = parent_path + '/plotting_data/enter_number_coordinates_' + DATASET + '.txt'
    c = open(file_name, "w")
    file_name = parent_path + '/plotting_data/enter_value_coordinates_' + DATASET + '.txt'
    d = open(file_name, "w")
    num_tw = NUM_TIME_WINDOWS
    for tw in range(num_tw+1):
        tw_df = enter_candidates.copy()
        tw_df = tw_df[tw_df['time window'] == tw]
        num_events = len(tw_df)
        values = tw_df['feature value'].tolist()
        value_sum = 0
        for v in values:
            value_sum = value_sum + int(v)
        #avoid division by 0
        if num_events == 0:
            average = 0
        else:
            average = value_sum/num_events
        text = 'tw ' + str(tw) + ': ' + str(num_events) + ', average value: ' + str(average) + '\n'
        f.write(text)
        text = str(tw) + ' ' + str(num_events) + '\n'
        c.write(text)
        text = str(tw) + ' ' + str(average) + '\n'
        d.write(text)
    c.close()
    d.close()
    f.write('\n')
    #exit candidates
    exit_candidates = candidates.copy()
    exit_candidates = exit_candidates[exit_candidates['feature type'] == 'exit']
    exit_candidates['feature value'] = exit_candidates['feature value'].apply(lambda x: np.nan if x == str(0) else x)
    exit_candidates.dropna(subset=['feature value'], inplace=True)
    num_exit = len(exit_candidates)
    text = 'Number of all exit event candidates: ' + str(num_exit) + '\n'
    f.write(text)
    text = 'candidates per time window:\n'
    f.write(text)
    file_name = parent_path + '/plotting_data/exit_number_coordinates_' + DATASET + '.txt'
    c = open(file_name, "w")
    file_name = parent_path + '/plotting_data/exit_value_coordinates_' + DATASET + '.txt'
    d = open(file_name, "w")
    for tw in range(num_tw+1):
        tw_df = exit_candidates.copy()
        tw_df = tw_df[tw_df['time window'] == tw]
        num_events = len(tw_df)
        values = tw_df['feature value'].tolist()
        value_sum = 0
        for v in values:
            value_sum = value_sum + int(v)
        #avoid division by 0
        if num_events == 0:
            average = 0
        else:
            average = value_sum/num_events
        text = 'tw ' + str(tw) + ': ' + str(num_events) + ', average value: ' + str(average) + '\n'
        f.write(text)
        text = str(tw) + ' ' + str(num_events) + '\n'
        c.write(text)
        text = str(tw) + ' ' + str(average) + '\n'
        d.write(text)
    c.close()
    d.close()
    f.write('\n')
    #load candidates
    load_candidates = candidates.copy()
    load_candidates = load_candidates[load_candidates['feature type'] == 'hl']
    num_load = len(load_candidates)
    text = 'Number of all load event candidates: ' + str(num_load) + '\n'
    f.write(text)
    text = 'candidates per time window:\n'
    f.write(text)
    file_name = parent_path + '/plotting_data/load_number_coordinates_' + DATASET + '.txt'
    c = open(file_name, "w")
    file_name = parent_path + '/plotting_data/load_value_coordinates_' + DATASET + '.txt'
    d = open(file_name, "w")
    for tw in range(num_tw+1):
        tw_df = load_candidates.copy()
        tw_df = tw_df[tw_df['time window'] == tw]
        num_events = len(tw_df)
        values = tw_df['feature value'].tolist()
        value_sum = 0
        for v in values:
            value_sum = value_sum + int(v)
        #avoid division by 0
        if num_events == 0:
            average = 0
        else:
            average = value_sum/num_events
        text = 'tw ' + str(tw) + ': ' + str(num_events) + ', average value: ' + str(average) + '\n'
        f.write(text)
        text = str(tw) + ' ' + str(num_events) + '\n'
        c.write(text)
        text = str(tw) + ' ' + str(average) + '\n'
        d.write(text)
    c.close()
    d.close()
    f.write('\n')
    #demand candidates
    demand_candidates = candidates.copy()
    demand_candidates = demand_candidates[demand_candidates['feature type'] == 'hd']
    demand_candidates['feature value'] = demand_candidates['feature value'].apply(lambda x: np.nan if x == str(0) else x)
    demand_candidates.dropna(subset=['feature value'], inplace=True)
    num_demand = len(demand_candidates)
    text = 'Number of all demand event candidates: ' + str(num_demand) + '\n'
    f.write(text)
    text = 'candidates per time window:\n'
    f.write(text)
    file_name = parent_path + '/plotting_data/demand_number_coordinates_' + DATASET + '.txt'
    c = open(file_name, "w")
    file_name = parent_path + '/plotting_data/demand_value_coordinates_' + DATASET + '.txt'
    d = open(file_name, "w")
    for tw in range(num_tw+1):
        tw_df = demand_candidates.copy()
        tw_df = tw_df[tw_df['time window'] == tw]
        num_events = len(tw_df)
        values = tw_df['feature value'].tolist()
        value_sum = 0
        for v in values:
            value_sum = value_sum + int(v)
        if num_events == 0:
            average = 0
        else:
            average = value_sum/num_events
        text = 'tw ' + str(tw) + ': ' + str(num_events) + ', average value: ' + str(average) + '\n'
        f.write(text)
        text = str(tw) + ' ' + str(num_events) + '\n'
        c.write(text)
        text = str(tw) + ' ' + str(average) + '\n'
        d.write(text)
    c.close()
    d.close()
    f.write('\n')
    #split candidates
    split_candidates = candidates.copy()
    split_candidates = split_candidates[split_candidates['feature type'] == 's']
    split_candidates['related object'] = convert_string_to_list(split_candidates['related object'])
    split_candidates = filter_split_join_events(split_candidates, NUM_TIME_WINDOWS, 's')
    num_split = len(split_candidates)
    text = 'Number of all split event candidates: ' + str(num_split) + '\n'
    f.write(text)
    text = 'candidates per time window:\n'
    f.write(text)
    file_name = parent_path + '/plotting_data/split_number_coordinates_' + DATASET + '.txt'
    c = open(file_name, "w")
    file_name = parent_path + '/plotting_data/split_value_coordinates_' + DATASET + '.txt'
    d = open(file_name, "w")
    for tw in range(num_tw+1):
        tw_df = split_candidates.copy()
        tw_df = tw_df[tw_df['time window'] == tw]
        num_events = len(tw_df)
        values = tw_df['feature value'].tolist()
        value_sum = 0
        for v in values:
            value_sum = value_sum + int(v)
        if num_events == 0:
            average = 0
        else:
            average = value_sum/num_events
        text = 'tw ' + str(tw) + ': ' + str(num_events) + ', average value: ' + str(average) + '\n'
        f.write(text)
        text = str(tw) + ' ' + str(num_events) + '\n'
        c.write(text)
        text = str(tw) + ' ' + str(average) + '\n'
        d.write(text)
    c.close()
    d.close()
    f.write('\n')
    #join candidates
    join_candidates = candidates.copy()
    join_candidates = join_candidates[join_candidates['feature type'] == 'j']
    join_candidates['related object'] = convert_string_to_list(join_candidates['related object'])
    join_candidates = filter_split_join_events(join_candidates, NUM_TIME_WINDOWS, 'j')
    num_join = len(join_candidates)
    text = 'Number of all join event candidates: ' + str(num_join) + '\n'
    f.write(text)
    text = 'candidates per time window:\n'
    f.write(text)
    file_name = parent_path + '/plotting_data/join_number_coordinates_' + DATASET + '.txt'
    c = open(file_name, "w")
    file_name = parent_path + '/plotting_data/join_value_coordinates_' + DATASET + '.txt'
    d = open(file_name, "w")
    for tw in range(num_tw+1):
        tw_df = join_candidates.copy()
        tw_df = tw_df[tw_df['time window'] == tw]
        num_events = len(tw_df)
        values = tw_df['feature value'].tolist()
        value_sum = 0
        for v in values:
            value_sum = value_sum + int(v)
        if num_events == 0:
            average = 0
        else:
            average = value_sum/num_events
        text = 'tw ' + str(tw) + ': ' + str(num_events) + ', average value: ' + str(average) + '\n'
        f.write(text)
        text = str(tw) + ' ' + str(num_events) + '\n'
        c.write(text)
        text = str(tw) + ' ' + str(average) + '\n'
        d.write(text)
    c.close()
    d.close()
    f.write('\n')
    #syn candidates
    syn_candidates = candidates.copy()
    syn_candidates = syn_candidates[syn_candidates['feature type'] == 'syn']
    syn_candidates['related object'] = convert_string_to_list(syn_candidates['related object'])
    syn_candidates = filter_syn_events(syn_candidates, NUM_TIME_WINDOWS)
    num_syn = len(syn_candidates)
    text = 'Number of all syn event candidates: ' + str(num_syn) + '\n'
    f.write(text)
    text = 'candidates per time window:\n'
    f.write(text)
    file_name = parent_path + '/plotting_data/syn_number_coordinates_' + DATASET + '.txt'
    c = open(file_name, "w")
    file_name = parent_path + '/plotting_data/syn_value_coordinates_' + DATASET + '.txt'
    d = open(file_name, "w")
    for tw in range(num_tw+1):
        tw_df = syn_candidates.copy()
        tw_df = tw_df[tw_df['time window'] == tw]
        num_events = len(tw_df)
        values = tw_df['feature value'].tolist()
        values.sort()
        if num_events == 0:
            average = 0
        else:
            average = values[int(num_events/2)]
        text = 'tw ' + str(tw) + ': ' + str(num_events) + ', average value: ' + str(average) + '\n'
        f.write(text)
        text = str(tw) + ' ' + str(num_events) + '\n'
        c.write(text)
        text = str(tw) + ' ' + str(average) + '\n'
        d.write(text)
    c.close()
    d.close()
    f.write('\n')
    total_event_candidates = num_enter + num_exit + num_demand + num_load + num_split + num_join + num_syn
    text = 'Total number of event candidates: ' + str(total_event_candidates)
    f.write(text)
    f.close()

def start_with_relations():
    #read relations
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '/final_relations/final_relations_' + DATASET + '.csv'
    #str_name = parent_path + '/final_relations_ocel2-p2p.csv'
    relations = pd.read_csv(str_name)

    #read hl events
    str_name = parent_path + '/hl_events_filtered/filtered_hl_events_' + DATASET + '.csv'
    #str_name = parent_path + '/filtered_hl_events_ocel2-p2p.csv'
    hl_events = pd.read_csv(str_name)

    relation_df = get_feature_types(relations, hl_events)
    hl_event_types = ['enter', 'exit', 's', 'j', 'syn', 'hl', 'hd']
    get_relation_summary(relation_df, hl_event_types)

def create_hl_event_stat():
    #read filtered events
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '/hl_events_filtered/filtered_hl_events_' + DATASET + '.csv'
    df = pd.read_csv(str_name)
    event_types = ['hl', 'hd', 's', 'j', 'syn', 'enter', 'exit']
    total_events = 0
    for t in event_types:
        tmp_df = df.copy()
        tmp_df = tmp_df[tmp_df['feature type']==t]
        num_events = len(tmp_df)
        total_events = total_events + num_events
        print(t + ': ' + str(num_events))
        #compute activities
        txt_name = parent_path + '/hl_event_activities/activity_summary_' + t + '_' + DATASET + '.txt'
        txt = open(txt_name, 'w')
        plot_data_name = parent_path + '/plotting_data/activity_summary_' + t + '_' + DATASET + '.txt'
        p = open(plot_data_name, 'w')
        if t=='hl':
            activities_position = tmp_df['related object'].tolist()
            txt.write('number of different activities for type : ' + t + ': ' + str(len(set(activities_position))) + '\n')
            count_act = Counter(activities_position)
            count = 0
            for key in list(count_act.keys()):
                count = count +1
                txt.write(key + ': ' + str(count_act[key]) + '\n')
                p.write(str(count_act[key])+ ' ' + str(count) + '\n')
                #print(str(key) + ',')
        if t=='hd':
            activities_position = tmp_df['related object'].tolist()
            real_activities_position = []
            for a in activities_position:
                ot = get_ot(a)
                real_activities_position.append(ot)
            txt.write('number of different activities: for type : ' + t + ': ' + str(len(set(real_activities_position))) + '\n')
            count_act = Counter(real_activities_position)
            count = 0
            for key in list(count_act.keys()):
                count = count +1
                txt.write(key + ': ' + str(count_act[key]) + '\n')
                p.write(str(count_act[key])+ ' ' + str(count) + '\n')
        if t=='s':
            activities_position = tmp_df['seg act1'].tolist()
            txt.write('number of different activities: for type : ' + t + ': ' + str(len(set(activities_position))) + '\n')
            count_act = Counter(activities_position)
            count = 0
            for key in list(count_act.keys()):
                count = count +1
                txt.write(key + ': ' + str(count_act[key]) + '\n')
                p.write(str(count_act[key])+ ' ' + str(count) + '\n')
        if t=='j' or t=='syn':
            activities_position = tmp_df['seg act2'].tolist()
            txt.write('number of different activities: for type : ' + t + ': ' + str(len(set(activities_position))) + '\n')
            count_act = Counter(activities_position)
            count = 0
            for key in list(count_act.keys()):
                count = count +1
                txt.write(key + ': ' + str(count_act[key]) + '\n')
                p.write(str(count_act[key])+ ' ' + str(count) + '\n')
        if t=='enter' or t=='exit':
            activities_position1 = tmp_df['seg act1'].tolist()
            activities_position2 = tmp_df['seg act2'].tolist()
            activities_position = []
            for i in range(len(activities_position1)):
                activities_position.append((activities_position1[i], activities_position2[i]))
            txt.write('number of different activities: for type : ' + t + ': ' + str(len(set(activities_position))) + '\n')
            count_act = Counter(activities_position)
            count = 0
            for key in list(count_act.keys()):
                count = count +1
                txt.write(str(key) + ': ' + str(count_act[key]) + '\n')
                p.write(str(count_act[key])+ ' ' + str(count) + '\n')
            if t=='enter':
                enter_positions_counter = count_act.copy()            
        txt.close()
        p.close()
    txt_name = parent_path + '/hl_event_activities/enter_exit_activity_comparison_.txt'
    txt = open(txt_name, 'w')    
    for key in list(count_act.keys()):
        txt.write(str(key) + ':\n')
        txt.write('enter: ' + str(enter_positions_counter[key]) + '\n')
        txt.write('exit: ' + str(count_act[key]) + '\n')
    print('total events: ' + str(total_events))

def examine_propagation():
    #read relations
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '/final_relations/final_relations_' +DATASET + '.csv'
    relations = pd.read_csv(str_name)
    #read events
    str_name = parent_path + '/hl_events_filtered/filtered_hl_events_' + DATASET + '.csv'
    events = pd.read_csv(str_name)

    all_events_in_rel = set(relations['eid1'].tolist())
    all_events_in_rel = all_events_in_rel.union(set(relations['eid2'].tolist()))

    eid_feature_type_dict = dict()
    for index,row in events.iterrows():
        eid = row['hl:eid']
        ft = row['feature type']
        if eid in all_events_in_rel:
            eid_feature_type_dict[eid] = ft

    type1 = []
    type2 = []
    for index,row in relations.iterrows():
        type1.append(eid_feature_type_dict[row['eid1']])
        type2.append(eid_feature_type_dict[row['eid2']])
    relations['type1'] = type1
    relations['type2'] = type2

    txt_name = parent_path + '/relation_stat/type_summary_' + DATASET + '.txt'
    txt = open(txt_name, "w")
    feature_types = ['hl', 'hd', 's', 'j', 'syn', 'enter', 'exit']
    for t in feature_types:
        t_rel = relations.copy()
        t_rel = t_rel[t_rel['type1']==t]
        txt.write('propagation from ' + str(t) + ': ' + str(len(t_rel)) + '\n')
        txt.write('to: \n')
        for t2 in feature_types:
            t2_rel = t_rel.copy()
            t2_rel = t2_rel[t2_rel['type2']==t2]
            txt.write(str(t2) + ': ' + str(len(t2_rel)) + '\n')
    txt.close()

    txt_name = parent_path + '/relation_stat/event_relation_counts_' + DATASET + '.txt'
    txt = open(txt_name, "w")
    txt_name = parent_path + '/plotting_data/relations_to_event_' + DATASET + '.txt'
    to_e = open(txt_name, 'w')
    txt_name = parent_path + '/plotting_data/relations_from_event_' + DATASET + '.txt'
    from_e = open(txt_name, 'w')
    count = 0
    for eid in all_events_in_rel:
        count = count+1
        eid_rel = relations.copy()
        eid_rel = eid_rel[eid_rel['eid1']==eid]
        txt.write('propagation from ' + str(eid) + ': ' + str(len(eid_rel)) + '\n')
        from_e.write(str(count) + ' ' + str(len(eid_rel)) + '\n')
        eid2_rel = relations.copy()
        eid2_rel = eid2_rel[eid2_rel['eid2']==eid]
        txt.write('propagation to ' + str(eid) + ': ' + str(len(eid2_rel)) + '\n')
        to_e.write(str(count) + ' ' + str(len(eid2_rel)) + '\n')
    txt.close()
    to_e.close()
    from_e.close()

def create_episodes():
    #read relations
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '/final_relations/final_relations_' + DATASET + '.csv'
    relations = pd.read_csv(str_name)
    #read events
    str_name = parent_path + '/hl_events_filtered/filtered_hl_events_' + DATASET + '.csv'
    events = pd.read_csv(str_name)

    #nodes of a graph
    all_events_in_rel = set(relations['eid1'].tolist())
    all_events_in_rel = all_events_in_rel.union(set(relations['eid2'].tolist()))
    
    #adjacent lists and edges
    rel_dict = dict()
    edges = []
    for eid in all_events_in_rel:
        eid_rel = relations.copy()
        eid_rel = eid_rel[eid_rel['eid1']==eid]
        succ = eid_rel['eid2'].tolist()
        rel_dict[eid] = succ
        for s in succ:
            e = (eid, s)
            edges.append(e)

    #compute graph, node=events, edges=relations
    g = nx.Graph()
    g.add_nodes_from(all_events_in_rel)
    g.add_edges_from(edges)
    #compute connected components = cascades
    conn_comp = nx.connected_components(g)
    #one subgraph = one cascade
    subgraphs = [g.subgraph(c).copy() for c in conn_comp]
    #subgraph dict contains list of nodes, list of edges and dict with labels per node
    subgraph_dict = dict()
    subgraph_number = dict()
    subgraph_count = 0
    length_of_episode = []
    for s in subgraphs:
        length_of_episode.append(len(s.nodes()))
        node_labels = examine_subgraph(s, events)
        #fix edges, necessary because of graph instead of digraph
        nodes = s.nodes()
        edges = []
        for n in nodes:
            n_succ = rel_dict[n]
            for succ in n_succ:
                e = (n, succ)
                edges.append(e)
        subgraph_dict[s] = [nodes, edges, node_labels]
        subgraph_number[s] = subgraph_count
        subgraph_count = subgraph_count + 1
    length_count = Counter(length_of_episode)
    open_path = parent_path + '/plotting_data/length_of_episodes_' + DATASET + '.txt'
    f = open(open_path, "w")
    for key in sorted(length_count.keys()):
        f.write(str(key) + ' ' + str(length_count[key]) + '\n')
    f.close()

    #compute matchings
    s1_count = 0
    #mathcing dict contains pairs of subgraphs as keys and their matching as value
    subgraphs_matchings = dict()
    #contains number of matching other subgrphs per subgraph
    number_matchings = dict()
    for s1 in subgraphs:
        for s2 in subgraphs:
            if subgraph_number[s1]!=subgraph_number[s2]:
                if set(subgraph_dict[s1][2].values())==set(subgraph_dict[s2][2].values()):
                    s1_s2_matchings = examine_edges(subgraph_dict[s1], subgraph_dict[s2])
                    if s1_s2_matchings != []:
                        s1_count = s1_count + 1
                        s_tuple = (s1, s2)
                        subgraphs_matchings[s_tuple] = s1_s2_matchings[0]
        number_matchings[s1] = s1_count
        s1_count = 0
    
    #save distribution of number of matchings
    open_path = parent_path + '/episodes_stat/distribution_of_matchings_' + DATASET + '.txt'
    f = open(open_path, "w")
    f.write('total number of episodes: ' + str(len(subgraphs)) + '\n')
    #find total number of different episodes
    found_subgraphs = []
    number_diff_episodes = 0
    unique_subgraphs = []
    for s in subgraphs:
        if s not in found_subgraphs:
            found_subgraphs.append(s)
            unique_subgraphs.append(s)
            number_diff_episodes = number_diff_episodes + 1
        if number_matchings[s]>0:
            f.write('number of subgraph that are isomorph to subgraph ' + str(subgraph_number[s]) + ': '+ str(number_matchings[s]) + '\n')
            f.write('these subgraphs are: ')
            for iso_tuple in list(subgraphs_matchings.keys()):
                if iso_tuple[0] == s:
                    found_subgraphs.append(iso_tuple[1])
                    f.write(str(subgraph_number[iso_tuple[1]]) + ', ')
            f.write('\n')
    f.write('total number of different episodes: ' + str(number_diff_episodes))
    f.close()

    open_path = parent_path + '/episodes_stat/frequency_of_paths_' + DATASET + '.txt'
    f = open(open_path, "w")
    open_path = parent_path + '/plotting_data/frequency_of_paths_' + DATASET + '.txt'
    g = open(open_path, "w")
    open_path = parent_path + '/episodes_stat/frequency_of_paths_sorted_' + DATASET + '.txt'
    h = open(open_path, "w")
    unique_list = []
    len_3_or_more_set = set()
    for s in unique_subgraphs:
        num = subgraph_number[s]
        nodes = subgraph_dict[s][0]
        for n in nodes:
            n_succ = rel_dict[n]
            for succ in n_succ:
                second_succ = rel_dict[succ]
                if second_succ!=[]:
                    len_3_or_more_set.add(num)
        for i in range(number_matchings[s]):
            unique_list.append(num)
    count_unique = Counter(unique_list)
    count = 0
    for key, value in count_unique.items():
        if value>=15:
            h.write(str(key) + ' ' + str(count_unique[key]+1) + '\n')
    for key in sorted(count_unique.keys()):
        count = count + 1
        f.write(str(key) + ' ' + str(count_unique[key]+1) + '\n')
        g.write(str(count) + ' ' + str(count_unique[key]+1)+ '\n')
    f.close()
    g.close()
    h.close()

    #save subgraphs
    for s in unique_subgraphs:
        if subgraph_number[s]>680:
        #if True:
            di_s = nx.DiGraph()
            di_s.add_nodes_from(subgraph_dict[s][0])
            s_edges = []
            for node in subgraph_dict[s][0]:
                n_succ = rel_dict[node]
                for succ in n_succ: 
                    if succ in subgraph_dict[s][0]:
                        s_edges.append((node,succ))
            di_s.add_edges_from(s_edges)
            fig = plt.figure()
            #pos=graphviz_layout(di_s, prog='dot')
            nx.draw_circular(di_s, labels = subgraph_dict[s][2], with_labels=True, font_size=8)
            #plt.show()
            save_name = parent_path+ '/episode_graphs_' + DATASET + '/graph'+ str(subgraph_number[s]) + '.png'
            fig.savefig(save_name)

def examine_edges(dict1, dict2):
    nodes1 = dict1[0]
    edges1 = dict1[1]
    label_dict1 = dict1[2]
    nodes2 = dict2[0]
    edges2 = dict2[1]
    label_dict2 = dict2[2]
    all_matchings = []
    if len(nodes1) == len(nodes2) and len(edges1)==len(edges2):
        sub1 = nx.DiGraph()
        sub1.add_nodes_from(nodes1)
        sub1.add_edges_from(edges1)
        sub2 = nx.DiGraph()
        sub2.add_nodes_from(nodes2)
        sub2.add_edges_from(edges2)
        is_iso = nx.is_isomorphic(sub1, sub2)
        if is_iso:
            for node in nodes1:
                sub1.nodes[node]['label'] = label_dict1[node]
            for node in nodes2:
                sub2.nodes[node]['label'] = label_dict2[node]
            matching = nx.vf2pp_isomorphism(sub1, sub2, node_label='label')
            if matching != None:
                all_matchings.append(matching)
    return all_matchings
        
def examine_subgraph(graph, events):
    nodes = list(graph.nodes)
    #edges = list(graph.edges)
    node_labels = dict()
    demand_count = 0
    for node in nodes:
        n_event = get_event_of_node(node, events)
        event_type = n_event['feature type'].tolist()[0]
        if event_type == 'enter':
            new_node_label = create_enter_exit_node(n_event, 'enter, ')
        if event_type == 'exit':
            new_node_label = create_enter_exit_node(n_event, 'exit, ')
        if event_type == 's':
            new_node_label = create_split_node(n_event)
        if event_type == 'j':
            new_node_label = create_join_syn_node(n_event, 'join, ')
        if event_type == 'syn':
            new_node_label = create_join_syn_node(n_event, 'syn, ')
        if event_type == 'hl':
            new_node_label = create_hl_node(n_event)
        if event_type == 'hd':
            demand_count = demand_count + 1
            new_node_label = create_hd_node(n_event)
        node_labels[node] = new_node_label
    #nx.draw(graph, labels= node_labels, with_labels = True)
    #plt.show()
    return node_labels

def create_hd_node(event):
    obj = event['related object'].tolist()[0]
    #workaround because i dont safe the object type
    ot = get_ot(obj)
    label = 'demand, ' + ot
    return label

def get_ot(input_sting):
    if DATASET=='order-management2':
        if 'i-' in input_sting:
            output = 'items'
        if 'o-' in input_sting:
            output = 'orders'
        if 'p-' in input_sting:
            output = 'packages'
    if DATASET == 'ocel2-p2p':
        if 'payment' in input_sting:
            output = 'payment'
        if 'invoice' in input_sting:
            output = 'invoice_receipt'
        if 'goods' in input_sting:
            output = 'goods_receipt'
        if 'purchase_order' in input_sting:
            output = 'purchase_order'
        if 'requisition' in input_sting:
            output = 'purchase_requisition'
        if 'material' in input_sting:
            output = 'material'
        if 'quotation' in input_sting:
            output = 'quotation'
        return output

def create_hl_node(event):
    obj = event['related object'].tolist()[0]
    label = 'load, ' + obj
    return label

def create_join_syn_node(event, type_string):
    act2 = event['seg act2'].tolist()[0]
    label = type_string + act2
    return label

def create_split_node(event):
    act1 = event['seg act1'].tolist()[0]
    label = 'split, ' + act1
    return label

def create_enter_exit_node(event, type_string):
    act1 = event['seg act1'].tolist()[0]
    act2 = event['seg act2'].tolist()[0]
    seg = (act1, act2)
    label = type_string + str(seg)
    return label

def get_event_of_node(node, events):
    working_events = events.copy()
    working_events = working_events[working_events['hl:eid']==node]
    return (working_events)

def examine_event_prop_count():
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    #str_name = parent_path + '/plotting_data/relations_from_event_' +DATASET + '.txt'
    str_name = parent_path + '/plotting_data/relations_to_event_' +DATASET + '.txt'
    with open(str_name) as f:
        lines = f.readlines()

    numbers = []
    for line in lines:
        line = line.strip()
        coordinates = line.split(' ')
        numbers.append(int(coordinates[1]))

    counts = Counter(numbers)
    print(max(counts.keys()))
    print(max(counts.values()))

    #txt_name = parent_path + '/plotting_data/count_from_event_' +DATASET + '.txt'
    txt_name = parent_path + '/plotting_data/count_to_event_' +DATASET + '.txt'
    txt = open(txt_name, "w")
    #txt = open("count_to_event.txt", "w")
    for number in sorted(counts.keys()):
        txt.write(str(number) + ' ' + str(counts[number]) + '\n')
    txt.close()

def test_relations():
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '/final_relations/final_relations_' + DATASET + '.csv'
    relations = pd.read_csv(str_name)
    print(relations.value_counts())
    print(len(relations))
    relations.drop_duplicates(inplace=True)
    print(len(relations))

def inspect_original_log():
    #define data set and paths
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    #dataset = 'order-management2'
    #dataset = 'ocel2-p2p'
    path = parent_path + './input_data/' + DATASET + '.sqlite'
    print(path)
    #read log and dataframe
    log, df = read_input(path)
    dfg = pm4py.ocel.discover_ocdfg(log)
    dot = pm4py.visualization.ocel.ocdfg.visualizer.apply(dfg)
    dot.render(directory='dfg_test')
    petrinet = pm4py.ocel.discover_oc_petri_net(log)

    #sort by timestamp
    df.sort_values(by='ocel:timestamp', inplace=True)
    print(df[:20][['ocel:activity', 'ocel:type:orders', 'ocel:type:items']])
    
def compute_relation_activities_summary():
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '/final_relations/final_relations_' + DATASET + '.csv'
    relations = pd.read_csv(str_name)
    str_name = parent_path + '/hl_events_filtered/filtered_hl_events_' + DATASET + '.csv'
    events = pd.read_csv(str_name)

    act_dict = dict()
    demand_ids = []
    for index,row in events.iterrows():
        id = row['hl:eid']
        ft = row['feature type']
        if ft=='hl':
            loc = row['related object']
            act = (ft, loc)
        if ft=='hd':
            demand_ids.append(id)
            loc = get_ot(row['related object'])
            act = (ft, loc)
        if ft=='s':
            loc = row['seg act1']
            act = (ft, loc)
        if ft=='j' or ft=='syn':
            loc = row['seg act2']
            act = (ft, loc)
        if ft=='enter' or ft=='exit':
            act1 = row['seg act1']
            act2 = row['seg act2']
            loc = (act1, act2)
            act = (ft, loc)
        act_dict[id] = act    
    print('len of demand ids: ' + str(len(demand_ids)))
    print(len(set(demand_ids)))

    act_rel = []
    count = 0
    for index, row in relations.iterrows():
        id1 = row['eid1']
        id2 = row['eid2']
        print(id2)
        if id1 in demand_ids and id2 in demand_ids:
            print(str(id1) + ', ' + str(id2))
            count = count +1
        act1 = act_dict[id1]
        act2 = act_dict[id2]
        rel = (act1, act2)
        act_rel.append(rel)
    print('num of demand rel: ' + str(count))

    print(len(act_rel))
    count_act_rel = Counter(act_rel)
    print(len(list(count_act_rel.keys())))
    open_path = parent_path + '/relation_stat/act_summary_' + DATASET + '.txt'
    f = open(open_path, "w")
    open_path = parent_path + '/relation_stat/act_summary_lowest_' + DATASET + '.txt'
    h = open(open_path, "w")
    f.write('total number of different activity prop: ' + str(len(list(count_act_rel.keys()))) + '\n')
    for key in list(count_act_rel.keys()):
        if count_act_rel[key]<=22:
            h.write(str(key) + ' ' + str(count_act_rel[key]) + '\n')
        f.write(str(key) + ' ' + str(count_act_rel[key]) + '\n')
    f.close()

def create_frequency_of_path_stat():
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    str_name = parent_path + '/plotting_data/frequency_of_paths_' +DATASET + '.txt'
    with open(str_name) as f:
        lines = f.readlines()

    frequency = []
    for line in lines:
        line = line.strip()
        coordinates = line.split(' ')
        frequency.append(int(coordinates[1]))
    sorted_frequency = reversed(sorted(frequency))

    open_path = parent_path + '/plotting_data/cascade_variant_frequency_' + DATASET + '.txt'
    f = open(open_path, "w")
    count = 0
    for i in sorted_frequency:
        f.write(str(count)+ ' ' + str(i) + '\n')
        count = count + 1
    f.close()

    count_frequency = Counter(frequency)
    od = collections.OrderedDict(sorted(count_frequency.items()))
    print(od)
    open_path = parent_path + '/plotting_data/cascade_variant_frequency_histogram_' + DATASET + '.txt'
    f = open(open_path, "w")
    for key in od.keys():
        f.write(str(key)+ ' ' + str(od[key])+ '\n')
    f.close()

    one_to_five = 0
    six_to_ten = 0
    eleven_to_fifteen = 0
    sixteen_to_twenty = 0
    twenty_to_twentyfive = 0
    twentysix_to_thirty = 0
    for key in od.keys():
        if key < 6:
            one_to_five = one_to_five + od[key]
        if key > 5 and key <11:
            six_to_ten = six_to_ten + od[key]
        if key >10 and key <16:
            eleven_to_fifteen = eleven_to_fifteen + od[key]
        if key>15 and key <21:
            sixteen_to_twenty = sixteen_to_twenty + od[key]
        if key>20 and key <26:
            twenty_to_twentyfive = twenty_to_twentyfive + od[key]
        if key>25 and key <31:
            twentysix_to_thirty = twentysix_to_thirty + od[key]
    open_path = parent_path + '/plotting_data/cascade_variant_frequency_histogram_clustered_' + DATASET + '.txt'
    f = open(open_path, "w")
    f.write('1 ' + str(one_to_five)+ '\n')
    f.write('2 ' + str(six_to_ten)+ '\n')
    f.write('3 ' + str(eleven_to_fifteen)+ '\n')
    f.write('4 ' + str(sixteen_to_twenty)+ '\n')
    f.write('5 ' + str(twenty_to_twentyfive)+ '\n')
    f.write('6 ' + str(twentysix_to_thirty)+ '\n')
    f.close()

#the main() method computes a df of high-level event candidates
main()

#the start_with_log_csv() method computes high-level events and propagation between them for the candidates
start_with_log_csv()

#the start_with_filtered_log() method computes propagation between the computes high-level events
#start_with_filtered_log()

#all other methods compute some statistics of the computation
#start_with_relations()
#create_event_candidate_statistic()
#create_hl_event_stat()
#examine_propagation()
#create_episodes()
#examine_event_prop_count()
#test_relations()
#inspect_original_log()
#compute_relation_activities_summary()
#create_frequency_of_path_stat()