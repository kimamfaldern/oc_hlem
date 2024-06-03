import pm4py
import pandas as pd
import numpy as np
from collections import Counter
from ocpa.algo.enhancement.token_replay_based_performance import algorithm as performance_factory
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
from msticpy.vis import mp_pandas_plot
from bokeh.io import export_png

def main():
    log = pm4py.read_xes('output.xes')
    df = pm4py.convert_to_dataframe(log)

    #look for object patterns in activities
    #not for every time segment but over whole log
    all_activities = pm4py.get_event_attribute_values(log, "concept:name")
    for key in all_activities.keys():
        #define df only with one activity
        act_df = df.copy(deep=True)
        act_df['concept:name'] = act_df['concept:name'].apply(lambda x: x if x==key else np.nan)
        #TODO: check na before or use log filtering instead
        act_df.dropna(subset=['concept:name'], inplace=True)
        #one_act_log = pm4py.filter_event_attribute_values(log, "concept:name", [key], level="event", retain=True)
        search_patterns(act_df)

    #define time bounds
    min_time = min(df['time:timestamp'])
    max_time = max(df['time:timestamp'])
    time_window = max_time-min_time
    number_segments = 10
    time_segment = time_window/number_segments
    time_bounds = []
    print(min_time)
    print(max_time)

    for i in range(number_segments):
        time_bounds.append(min_time + (i+1)*time_segment)
    print(time_bounds)

    #partition df according to time bounds
    df.sort_values("time:timestamp", inplace=True)    
    tmp_df = pd.DataFrame()
    j = 0
    for index, row in df.iterrows():    #probably better to use apply or somesting
        if row["time:timestamp"] <= time_bounds[j]:
            tmp_df = pd.concat([tmp_df, row.to_frame().T])
        else:
            j=j+1
            #print(tmp_df)
            mine_hle(tmp_df, min_time, time_segment, j)
            tmp_df = pd.DataFrame()

def mine_hle(df, min_time, time_segment, curr_seg_count):
    #convert string of objects to list
    #map nans
    df.fillna('NaN', inplace=True)
    #split
    df['type:object:GDSRCPT'] = df['type:object:GDSRCPT'].apply(lambda x: [] if x == 'NaN' else x.split('\''))
    df['type:object:INVOICE'] = df['type:object:INVOICE'].apply(lambda x: [] if x == 'NaN' else x.split('\''))
    df['type:object:MATERIAL'] = df['type:object:MATERIAL'].apply(lambda x: [] if x == 'NaN' else x.split('\''))
    df['type:object:PURCHORD'] = df['type:object:PURCHORD'].apply(lambda x: [] if x == 'NaN' else x.split('\''))
    df['type:object:PURCHREQ'] = df['type:object:PURCHREQ'].apply(lambda x: [] if x == 'NaN' else x.split('\''))
    #remove trash
    df['type:object:GDSRCPT'] = df['type:object:GDSRCPT'].apply(lambda x: own_del(x))
    df['type:object:INVOICE'] = df['type:object:INVOICE'].apply(lambda x: own_del(x))
    df['type:object:MATERIAL'] = df['type:object:MATERIAL'].apply(lambda x: own_del(x))
    df['type:object:PURCHORD'] = df['type:object:PURCHORD'].apply(lambda x: own_del(x))
    df['type:object:PURCHREQ'] = df['type:object:PURCHREQ'].apply(lambda x: own_del(x))
    #count objects per event
    dict_df = pd.DataFrame()
    copy_df = df.copy()
    dict_df['dict:GDSRCPT'] = copy_df['type:object:GDSRCPT'].apply(lambda x: Counter(x))
    dict_df['dict:INVOICE'] = copy_df['type:object:INVOICE'].apply(lambda x: Counter(x))
    dict_df['dict:MATERIAL'] = copy_df['type:object:MATERIAL'].apply(lambda x: Counter(x))
    dict_df['dict:PURCHORD'] = copy_df['type:object:PURCHORD'].apply(lambda x: Counter(x))
    dict_df['dict:PURCHREQ'] = copy_df['type:object:PURCHREQ'].apply(lambda x: Counter(x))

    #count of objects within object type for whole df (time segment)
    GDSRCPT_dict = Counter({})
    INVOICE_dict = Counter({})
    MATERIAL_dict = Counter({})
    PURCHORD_dict = Counter({})
    PURCHREQ_dict = Counter({})
    for index, row in dict_df.iterrows():
        GDSRCPT_dict.update(row['dict:GDSRCPT'])
        INVOICE_dict.update(row['dict:INVOICE'])
        MATERIAL_dict.update(row['dict:MATERIAL'])
        PURCHORD_dict.update(row['dict:PURCHORD'])
        PURCHREQ_dict.update(row['dict:PURCHREQ'])
    
    #find objects with highest demand
    #initialize max_ with highest count 
    #initialieze list to collect all objects that have highest count
    max_GDSRCPT = GDSRCPT_dict[max(GDSRCPT_dict, key=GDSRCPT_dict.get)]
    max_GDSRCPT_list = []
    max_INVOICE = INVOICE_dict[max(INVOICE_dict, key=INVOICE_dict.get)]
    max_INVOICE_list = []
    max_MATERIAL = MATERIAL_dict[max(MATERIAL_dict, key=MATERIAL_dict.get)]
    max_MATERIAL_list = []
    max_PURCHORD = PURCHORD_dict[max(PURCHORD_dict, key=PURCHORD_dict.get)]
    max_PURCHORD_list = []
    max_PURCHREQ = PURCHREQ_dict[max(PURCHREQ_dict, key=PURCHREQ_dict.get)]
    max_PURCHREQ_list = []
    #collect all items which have the highest count
    for key, value in GDSRCPT_dict.items():
        if value==max_GDSRCPT:
            max_GDSRCPT_list.append(key)
    for key, value in INVOICE_dict.items():
        if value==max_INVOICE:
            max_INVOICE_list.append(key)
    for key, value in MATERIAL_dict.items():
        if value==max_MATERIAL:
            max_MATERIAL_list.append(key)
    for key, value in PURCHORD_dict.items():
        if value==max_PURCHORD:
            max_PURCHORD_list.append(key)
    for key, value in PURCHREQ_dict.items():
        if value==max_PURCHREQ:
            max_PURCHREQ_list.append(key)

    #initialize lists for high level log
    event = []
    frequency = []
    start_time = []
    end_time = []

    #add high level events related to high demand of one object to log
    for obj in max_GDSRCPT_list:
        event.append('high demand of GDSRCPT: ' + str(obj))
        frequency.append(max_GDSRCPT)
        start_time.append(min_time + (curr_seg_count-1)*time_segment)
        end_time.append(min_time + curr_seg_count*time_segment)
    for obj in max_INVOICE_list:
        event.append('high demand of INVOICE: ' + str(obj))
        frequency.append(max_INVOICE)
        start_time.append(min_time + (curr_seg_count-1)*time_segment)
        end_time.append(min_time + curr_seg_count*time_segment)
    for obj in max_MATERIAL_list:
        event.append('high demand of MATERIAL: ' + str(obj))
        frequency.append(max_MATERIAL)
        start_time.append(min_time + (curr_seg_count-1)*time_segment)
        end_time.append(min_time + curr_seg_count*time_segment)
    for obj in max_PURCHORD_list:
        event.append('high demand of PURCHORD: ' + str(obj))
        frequency.append(max_PURCHORD)
        start_time.append(min_time + (curr_seg_count-1)*time_segment)
        end_time.append(min_time + curr_seg_count*time_segment)
    for obj in max_PURCHREQ_list:
        event.append('high demand of PURCHREQ: ' + str(obj))
        frequency.append(max_PURCHREQ)
        start_time.append(min_time + (curr_seg_count-1)*time_segment)
        end_time.append(min_time + curr_seg_count*time_segment)

    #add high level events related to load of one object type to log
    event.append('high load of object type GDSRCPT')
    frequency.append(len(GDSRCPT_dict))
    start_time.append(min_time + (curr_seg_count-1)*time_segment)
    end_time.append(min_time + curr_seg_count*time_segment)
    event.append('high load of object type INVOICE')
    frequency.append(len(INVOICE_dict))
    start_time.append(min_time + (curr_seg_count-1)*time_segment)
    end_time.append(min_time + curr_seg_count*time_segment)
    event.append('high load of object type MATERIAL')
    frequency.append(len(MATERIAL_dict))
    start_time.append(min_time + (curr_seg_count-1)*time_segment)
    end_time.append(min_time + curr_seg_count*time_segment)
    event.append('high load of object type PURCHORD')
    frequency.append(len(PURCHORD_dict))
    start_time.append(min_time + (curr_seg_count-1)*time_segment)
    end_time.append(min_time + curr_seg_count*time_segment)
    event.append('high load of object type PURCHREQ')
    frequency.append(len(PURCHREQ_dict))
    start_time.append(min_time + (curr_seg_count-1)*time_segment)
    end_time.append(min_time + curr_seg_count*time_segment)
    
    #count different object types
    sum_column = []
    for index, row in dict_df.iterrows():
        sum_column.append(int(bool(row['dict:GDSRCPT'])) + int(bool(row['dict:INVOICE'])) + int(bool(row['dict:MATERIAL'])) + int(bool(row['dict:PURCHORD'])) + int(bool(row['dict:PURCHREQ'])))
    dict_df['num_object_types'] = sum_column
    
    max_object_types = max(sum_column)
    max_df = dict_df[dict_df.num_object_types==max_object_types]

    #add high level event of high load of object types to log
    event.append('high load (overall object types)')
    frequency.append(max_df.shape[0])
    start_time.append(min_time + (curr_seg_count-1)*time_segment)
    end_time.append(min_time + curr_seg_count*time_segment)

    #explore waiting times and split/joins
    cases = df['case:concept:name'].unique()
    for case in cases:
        case_df = df[df['case:concept:name']==case]
        #if events with materials happen at same time, merge them to one event to for further computations 
        case_df = preprocess_case_df(case_df)
        case_df_copy = case_df.copy()
        #sort by timestamp
        for index, row in case_df_copy.iterrows():
            successor = get_successor(row,index, case_df_copy)
            #HINT: complete df is spitted according to time bounds. This leads to different start/end points of cases. Therefor some splits or joins or not/wrong found.
            if(str(successor)!='NaN'):
                succ_act = set(successor['concept:name'])
                #check if object leavs the process
                if len(succ_act)==1:
                    #TODO: theoretical change set to list
                    curr_obj = set()
                    if row['type:object:GDSRCPT']!=[]:
                        curr_obj.add('GDSRCPT')
                    if row['type:object:INVOICE']!=[]:
                        curr_obj.add('INVOICE')
                    if row['type:object:MATERIAL']!=[]:
                        curr_obj.add('MATERIAL')
                    if row['type:object:PURCHORD']!=[]:
                        curr_obj.add('PURCHORD')
                    if row['type:object:PURCHREQ']!=[]:
                        curr_obj.add('PURCHREQ')
                    shared_obj = curr_obj.intersection(set(successor['type:object']))
                    for o in shared_obj:
                        curr_obj.remove(o)
                    if curr_obj!=set():
                        #add high level event of object type join to log
                        event.append('object type ' + str(curr_obj) + ' leaves the process in activity ' + str(row['concept:name']))
                        frequency.append(len(curr_obj))
                        #TODO: check times
                        start_time.append(min_time + (curr_seg_count-1)*time_segment)
                        end_time.append(min_time + curr_seg_count*time_segment)
                elif len(succ_act)>1:
                    join_df = successor.copy()
                    join_df = join_df.drop_duplicates(subset=['concept:name'])
                    #add high level event of object join to log
                    event.append('object split in activity ' + str(row['concept:name']) + ' with objects ' + str(join_df['object'].sum()))
                    frequency.append(len(join_df['object'].sum()))
                    #TODO: check times
                    start_time.append(min_time + (curr_seg_count-1)*time_segment)
                    end_time.append(min_time + curr_seg_count*time_segment) 
        for index, row in case_df.iterrows():
            predecessor, case_df = get_predecessor(row, index, case_df)
            if str(predecessor)!='NaN':
                #get execution time and predecessor time and index
                exec_time = row['time:timestamp']
                early_time = min(predecessor['time:timestamp'])
                early_index = predecessor[predecessor['time:timestamp']==early_time].index
                late_time = max(predecessor['time:timestamp'])
                late_index = predecessor[predecessor['time:timestamp']==early_time].index
                #lagging and delay time are based on opera measures but need to be calculated manually because of different log format
                lagging_time = late_time - early_time
                delay_time = exec_time - late_time
                #add high level event of high lagging time per activity to log
                event.append('high lagging time of object ' + str(predecessor.loc[early_index, 'object']) + ' for activity ' + str(row['concept:name']))
                frequency.append(lagging_time)
                start_time.append(min_time + (curr_seg_count-1)*time_segment)
                end_time.append(min_time + curr_seg_count*time_segment)
                #discover joins of object types
                pred_act = set(predecessor['concept:name'])
                #test if join happens
                #check if object type joins
                if len(pred_act)==1:
                    #get object types of current event
                    #TODO: change set to list
                    curr_obj = set()
                    if row['type:object:GDSRCPT']!=[]:
                        curr_obj.add('GDSRCPT')
                    if row['type:object:INVOICE']!=[]:
                        curr_obj.add('INVOICE')
                    if row['type:object:MATERIAL']!=[]:
                        curr_obj.add('MATERIAL')
                    if row['type:object:PURCHORD']!=[]:
                        curr_obj.add('PURCHORD')
                    if row['type:object:PURCHREQ']!=[]:
                        curr_obj.add('PURCHREQ')
                    shared_obj = curr_obj.intersection(set(predecessor['type:object']))
                    for o in shared_obj:
                        curr_obj.remove(o)
                    if curr_obj!=set():
                        #add high level event of object type join to log
                        event.append('object type ' + str(curr_obj) + ' joins the process in activity ' + str(row['concept:name']))
                        frequency.append(len(curr_obj))
                        #TODO: check times
                        start_time.append(min_time + (curr_seg_count-1)*time_segment)
                        end_time.append(min_time + curr_seg_count*time_segment)
                #check if multiple activities are joint
                elif len(pred_act)>1:
                    join_df = predecessor.copy()
                    join_df = join_df.drop_duplicates(subset=['concept:name'])
                    #add high level event of object join to log
                    event.append('object join in activity ' + str(row['concept:name']) + ' with objects ' + str(join_df['object'].sum()))
                    frequency.append(len(join_df['object'].sum()))
                    #TODO: check times
                    start_time.append(min_time + (curr_seg_count-1)*time_segment)
                    end_time.append(min_time + curr_seg_count*time_segment)                   

    #add high level event of high lagging time per activity to log
    event.append('high load (overall object types)')
    frequency.append(max_df.shape[0])
    start_time.append(min_time + (curr_seg_count-1)*time_segment)
    end_time.append(min_time + curr_seg_count*time_segment)

    #convert to dataframe
    d = {'event': event, 'frequency': frequency, 'start time': start_time, 'end time': end_time}
    hl_log = pd.DataFrame(data=d)
    print(hl_log)

def preprocess_case_df(df):
    num_times = len(set(df['time:timestamp']))
    if num_times != len(df.index):
        copy_df = df.copy()
        copy_df = copy_df.duplicated(subset=['time:timestamp'], keep=False)#, 'type:object:GDSRCPT', 'type:object:INVOICE', 'type:object:MATERIAL', 'type:object:PURCHORD', 'type:object:PURCHREQ'], keep=False)
        copy_df = copy_df[copy_df]
        c = 0
        for index in copy_df.index:
            if c>0:
                if (set(df.loc[index, 'type:object:GDSRCPT'])==set(df.loc[last_index, 'type:object:GDSRCPT']) and set(df.loc[index, 'type:object:INVOICE'])==set(df.loc[last_index, 'type:object:INVOICE']) and set(df.loc[index, 'type:object:MATERIAL'])==set(df.loc[last_index, 'type:object:MATERIAL']) and set(df.loc[index, 'type:object:PURCHORD'])==set(df.loc[last_index, 'type:object:PURCHORD']) and set(df.loc[index, 'type:object:PURCHREQ'])==set(df.loc[last_index, 'type:object:PURCHREQ'])):
                    df.at[last_index, 'concept:name'] = df.loc[last_index, 'concept:name'] + ", " + df.loc[index, 'concept:name']
                    df.drop(index, inplace=True)
                else:
                    last_index = index
            else:
                last_index = index
            c = c+1
        return df
    else:
        return df

#for event in row at index index and case in df return all direct predecessors in pred and the case without the seen objects in case_df
def get_predecessor(row, index, df):
    save_df = df.copy()
    if row['time:timestamp']==min(df['time:timestamp']):
        return ('NaN', df)
    else:
        act_time = row['time:timestamp']
        df.drop(index, inplace=True)
        copy_df = df.copy()
        df['intersect:GDSRCPT'] = copy_df['type:object:GDSRCPT'].apply(lambda x: list_intersect(x, 'type:object:GDSRCPT', row))
        df['intersect:INVOICE'] = copy_df['type:object:INVOICE'].apply(lambda x: list_intersect(x, 'type:object:INVOICE', row))
        df['intersect:MATERIAL'] = copy_df['type:object:MATERIAL'].apply(lambda x: list_intersect(x, 'type:object:MATERIAL', row))
        df['intersect:PURCHORD'] = copy_df['type:object:PURCHORD'].apply(lambda x: list_intersect(x, 'type:object:PURCHORD', row))
        df['intersect:PURCHREQ'] = copy_df['type:object:PURCHREQ'].apply(lambda x: list_intersect(x, 'type:object:PURCHREQ', row))
        time = []
        act = []
        obj_type = []
        obj = []
        for i, r in df.iterrows():
            if r['time:timestamp']<act_time:
                if r['intersect:GDSRCPT']!=[]:
                    #idea: remove objects from case df row which already intersected an event
                    #problem: object flow can split for one specific object
                    #remark: generell objects should be removed, but if events happen at same time -> problem
                    #solution: merge events with same object and timestamps to one event
                    #next problem: if objects are removed from event, they cant be found in predecessor search
                    #solution: shouldnt be a problem for non-circulat behaviour
                    #TODO: implement
                    new_GDSRCPT = save_df.loc[i, 'type:object:GDSRCPT']
                    for element in r['intersect:GDSRCPT']:
                        new_GDSRCPT.remove(element)
                    save_df.at[i, 'type:object:GDSRCPT'] = new_GDSRCPT
                    time.append(r['time:timestamp'])
                    act.append(r['concept:name'])
                    obj_type.append('GDSRCPT')
                    obj.append(r['intersect:GDSRCPT'])
                    #pred = pd.concat([pred, r.to_frame().T])
                if r['intersect:INVOICE']!=[]:
                    new_INVOICE = save_df.loc[i, 'type:object:INVOICE']
                    for element in r['intersect:INVOICE']:
                        new_INVOICE.remove(element)
                    save_df.at[i, 'type:object:INVOICE'] = new_INVOICE
                    time.append(r['time:timestamp'])
                    act.append(r['concept:name'])
                    obj_type.append('INVOICE')
                    obj.append(r['intersect:INVOICE'])
                    #pred = pd.concat([pred, r.to_frame().T])
                if r['intersect:MATERIAL']!=[]:
                    new_MATERIAL = save_df.loc[i, 'type:object:MATERIAL']
                    for element in r['intersect:MATERIAL']:
                        new_MATERIAL.remove(element)
                    save_df.at[i, 'type:object:MATERIAL'] = new_MATERIAL
                    time.append(r['time:timestamp'])
                    act.append(r['concept:name'])
                    obj_type.append('MATERIAL')
                    obj.append(r['intersect:MATERIAL'])
                    #pred = pd.concat([pred, r.to_frame().T])
                if r['intersect:PURCHORD']!=[]:
                    new_PURCHORD = save_df.loc[i, 'type:object:PURCHORD']
                    for element in r['intersect:PURCHORD']:
                        new_PURCHORD.remove(element)
                    save_df.at[i, 'type:object:PURCHORD'] = new_PURCHORD
                    time.append(r['time:timestamp'])
                    act.append(r['concept:name'])
                    obj_type.append('PURCHORD')
                    obj.append(r['intersect:PURCHORD'])
                    #pred = pd.concat([pred, r.to_frame().T])
                if r['intersect:PURCHREQ']!=[]:
                    new_PURCHREQ = save_df.loc[i, 'type:object:PURCHREQ']
                    for element in r['intersect:PURCHREQ']:
                        new_PURCHREQ.remove(element)
                    save_df.at[i, 'type:object:PURCHREQ'] = new_PURCHREQ
                    time.append(r['time:timestamp'])
                    act.append(r['concept:name'])
                    obj_type.append('PURCHREQ')
                    obj.append(r['intersect:PURCHREQ'])
                    #pred = pd.concat([pred, r.to_frame().T])
        if time == []:
            return 'NaN', save_df
        else:
            d = {'time:timestamp': time, 'concept:name': act, 'type:object': obj_type, 'object': obj}
            pred = pd.DataFrame(data=d)
            return pred, save_df
        
def get_successor(row, index, df):
    save_df = df.copy()
    if row['time:timestamp']==max(df['time:timestamp']): #possible bacause of preprocessing
        return 'NaN'
    else:
        act_time = row['time:timestamp']
        act_objects = row['type:object:GDSRCPT']
        act_objects.extend(row['type:object:INVOICE'])
        act_objects.extend(row['type:object:MATERIAL'])
        act_objects.extend(row['type:object:PURCHORD'])
        act_objects.extend(row['type:object:PURCHREQ'])
        save_df.drop(index, inplace=True)
        copy_df = df.copy()
        save_df['intersect:GDSRCPT'] = copy_df['type:object:GDSRCPT'].apply(lambda x: list_intersect(x, 'type:object:GDSRCPT', row))
        save_df['intersect:INVOICE'] = copy_df['type:object:INVOICE'].apply(lambda x: list_intersect(x, 'type:object:INVOICE', row))
        save_df['intersect:MATERIAL'] = copy_df['type:object:MATERIAL'].apply(lambda x: list_intersect(x, 'type:object:MATERIAL', row))
        save_df['intersect:PURCHORD'] = copy_df['type:object:PURCHORD'].apply(lambda x: list_intersect(x, 'type:object:PURCHORD', row))
        save_df['intersect:PURCHREQ'] = copy_df['type:object:PURCHREQ'].apply(lambda x: list_intersect(x, 'type:object:PURCHREQ', row))
        time = []
        act = []
        obj_type = []
        obj = []
        for i, r in save_df.iterrows():
            #if should be unnecessary
            if r['time:timestamp']>act_time:
                if r['intersect:GDSRCPT']!=[]:
                    is_succ = False
                    for element in r['intersect:GDSRCPT']:
                        is_in = element in act_objects
                        if is_in:
                            is_succ = True
                            act_objects.remove(element) 
                    if is_succ:
                        time.append(r['time:timestamp'])
                        act.append(r['concept:name'])
                        obj_type.append('GDSRCPT')
                        obj.append(r['intersect:GDSRCPT'])
                if r['intersect:INVOICE']!=[]:
                    is_succ = False
                    for element in r['intersect:INVOICE']:
                        is_in = element in act_objects
                        if is_in:
                            is_succ = True
                            act_objects.remove(element) 
                    if is_succ:
                        time.append(r['time:timestamp'])
                        act.append(r['concept:name'])
                        obj_type.append('INVOICE')
                        obj.append(r['intersect:INVOICE'])
                if r['intersect:MATERIAL']!=[]:
                    is_succ = False
                    for element in r['intersect:MATERIAL']:
                        is_in = element in act_objects
                        if is_in:
                            is_succ = True
                            act_objects.remove(element) 
                    if is_succ:
                        time.append(r['time:timestamp'])
                        act.append(r['concept:name'])
                        obj_type.append('MATERIAL')
                        obj.append(r['intersect:MATERIAL'])
                if r['intersect:PURCHORD']!=[]:
                    is_succ = False
                    for element in r['intersect:PURCHORD']:
                        is_in = element in act_objects
                        if is_in:
                            is_succ = True
                            act_objects.remove(element) 
                    if is_succ:
                        time.append(r['time:timestamp'])
                        act.append(r['concept:name'])
                        obj_type.append('PURCHORD')
                        obj.append(r['intersect:PURCHORD'])
                if r['intersect:PURCHREQ']!=[]:
                    is_succ = False
                    for element in r['intersect:PURCHREQ']:
                        is_in = element in act_objects
                        if is_in:
                            is_succ = True
                            act_objects.remove(element) 
                    if is_succ:
                        time.append(r['time:timestamp'])
                        act.append(r['concept:name'])
                        obj_type.append('PURCHREQ')
                        obj.append(r['intersect:PURCHREQ'])
        if time == []:
            return 'NaN'
        else:
            d = {'time:timestamp': time, 'concept:name': act, 'type:object': obj_type, 'object': obj}
            pred = pd.DataFrame(data=d)
            return pred

def search_patterns(df_in):
    df = df_in.copy()
    #convert string of objects to list
    #map nans
    df.fillna('NaN', inplace=True)
    #split
    df['type:object:GDSRCPT'] = df['type:object:GDSRCPT'].apply(lambda x: [] if x == 'NaN' else x.split('\''))
    df['type:object:INVOICE'] = df['type:object:INVOICE'].apply(lambda x: [] if x == 'NaN' else x.split('\''))
    df['type:object:MATERIAL'] = df['type:object:MATERIAL'].apply(lambda x: [] if x == 'NaN' else x.split('\''))
    df['type:object:PURCHORD'] = df['type:object:PURCHORD'].apply(lambda x: [] if x == 'NaN' else x.split('\''))
    df['type:object:PURCHREQ'] = df['type:object:PURCHREQ'].apply(lambda x: [] if x == 'NaN' else x.split('\''))
    #remove trash
    df['type:object:GDSRCPT'] = df['type:object:GDSRCPT'].apply(lambda x: own_del(x))
    df['type:object:INVOICE'] = df['type:object:INVOICE'].apply(lambda x: own_del(x))
    df['type:object:MATERIAL'] = df['type:object:MATERIAL'].apply(lambda x: own_del(x))
    df['type:object:PURCHORD'] = df['type:object:PURCHORD'].apply(lambda x: own_del(x))
    df['type:object:PURCHREQ'] = df['type:object:PURCHREQ'].apply(lambda x: own_del(x))
    #count objects
    df['type:object:GDSRCPT'] = df['type:object:GDSRCPT'].apply(lambda x: Counter(x))
    df['type:object:INVOICE'] = df['type:object:INVOICE'].apply(lambda x: Counter(x))
    df['type:object:MATERIAL'] = df['type:object:MATERIAL'].apply(lambda x: Counter(x))
    df['type:object:PURCHORD'] = df['type:object:PURCHORD'].apply(lambda x: Counter(x))
    df['type:object:PURCHREQ'] = df['type:object:PURCHREQ'].apply(lambda x: Counter(x))
    df['type:object:GDSRCPT'] = df['type:object:GDSRCPT'].apply(lambda x: sum(x.values()))
    df['type:object:INVOICE'] = df['type:object:INVOICE'].apply(lambda x: sum(x.values()))
    df['type:object:MATERIAL'] = df['type:object:MATERIAL'].apply(lambda x: sum(x.values()))
    df['type:object:PURCHORD'] = df['type:object:PURCHORD'].apply(lambda x: sum(x.values()))
    df['type:object:PURCHREQ'] = df['type:object:PURCHREQ'].apply(lambda x: sum(x.values()))
    #TODO: sort df by timestamp
    #keep first and last occurences of object patterns 
    #idea: see if patterns changes over time
    #TODO additionally: search for deviations
    pattern_firsts = df.drop_duplicates(subset=['type:object:GDSRCPT', 'type:object:INVOICE', 'type:object:MATERIAL', 'type:object:PURCHORD', 'type:object:PURCHREQ'], keep='first')
    pattern_lasts = df.drop_duplicates(subset=['type:object:GDSRCPT', 'type:object:INVOICE', 'type:object:MATERIAL', 'type:object:PURCHORD', 'type:object:PURCHREQ'], keep='last')
    
    print(pattern_firsts.to_string())
    print(pattern_lasts.to_string())
    #observation: in this dataset, only the number of involved materials really changes (invoice in two activites)
    #this is not really a deviation or a change of pattern over time
    #idea: maybe in a next step compute how many object types vary and set a threshold for deviations/patterns
    #also: there could be a pattern within the deviation, e.g in this dataset the different numbers of involved materials are often identical
    
    
    #plot timestamps in timeline to see pattern ranges
    #TODO: adapt plot and define high level events
    #plot = pattern_firsts.mp_plot.timeline(time_column="time:timestamp")
    #file_name = "plot.png"
    #export_png(plot, filename=file_name)

def own_del(x):
    del x[::2]
    return x

def list_intersect(x, s, row):
    return_list = []
    for i in row[s]:
        if i in x:
            return_list.append(i)
    return return_list


main()