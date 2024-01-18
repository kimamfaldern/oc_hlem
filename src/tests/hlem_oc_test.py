import pm4py
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from ocpa_test.ocpa.algo.util.process_executions import factory as process_executions_factory
from ocpa.objects.log.importer.ocel2.sqlite import factory as ocel_import_factory
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory_old
pd.options.mode.chained_assignment = None  # default='warn'


def main():
    dataset = 'order-management2'
    path = dataset + '.sqlite'
    log = pm4py.read_ocel2_sqlite(path)
    df = log.get_extended_table() #extended_df
    objects = pm4py.ocel_get_object_types(log)
    #set handling of object type: True=only type matters, not single instance, False=instance matters
    object_handling = {}
    if dataset == 'procure-to-pay':
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

    print(extended_df)
    qcids = []
    for index, row in df.iterrows():
        e = row['ocel:eid']
        qcids.append(eid_dict[e])
    print(qcids)
    df['ocel:process:execution'] = qcids

    print(df)
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

    #count all objects per object type per activity 
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
    number_segments = 428
    time_segment = time_window/number_segments
    time_bounds = []
    print(min_time)
    print(max_time)

    for i in range(number_segments):
        time_bounds.append(min_time + (i+1)*time_segment)
    print(time_bounds)

    #partition df according to time bounds
    counter_df.sort_values("ocel:timestamp", inplace=True)    
    tmp_df_dict = {}
    for j in range(number_segments-1):
        str_name = 'tmp_df' + str (j)
        tmp_df_dict[str_name] = counter_df[counter_df['ocel:timestamp']>time_bounds[j]]
        tmp_df_dict[str_name] = counter_df[counter_df['ocel:timestamp']<=time_bounds[j+1]]
        #print(tmp_df_dict[str_name])
        if dataset == 'order-management2':
            objects = ['products', 'orders', 'items', 'employees', 'customers', 'packages']
            object_handling['orders'] = True
            object_handling['items'] = True
            object_handling['packages'] = True
            object_handling['products'] = False
            object_handling['employees'] = False
            object_handling['customers'] = False
        if dataset == 'procure-to-pay':
            objects = ['goods receipt', 'invoice receipt', 'material', 'payment', 'purchase_order', 'purchase_requisition', 'quotation']
            object_handling = {'goods receipt': True, 'invoice receipt': True, 'material': True, 'payment': True, 'purchase_order':True, 'purchase_requisition': True, 'quotation': True}
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
            for instance in max_list_dict[str(obj)]:
                event.append('high demand of ' + str(obj) + ': '+ str(instance))
                event_type.append('hd')
                frequency.append(max_dict[str(obj)])
                start_time.append(min_time + (curr_seg_count-1)*time_segment)
                end_time.append(min_time + curr_seg_count*time_segment)
                time_frame.append(curr_seg_count)
                qcid.append(np.nan)
                related_object.append(instance)
    #add high level events related to load of one object type to log (only occurence of object type counts)
    for obj in objects:
        if object_handling[obj]:
            event.append('high load of object type ' + str(obj))
            event_type.append('hl')
            frequency.append(len(count_dict[str(obj)]))
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
    print('cases:')
    print(cases)
    print(len(cases))
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
                        remaining_obj = objects
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
                    joining_obj_types = join_df['ocel:type'].unique()
                    join_df = join_df.drop_duplicates(subset=['ocel:activity'])
                    #if instance matters for one involved object type, take instances for all object types
                    handling = True
                    for obj in joining_obj_types:
                        if object_handling[obj]==False:
                            handling = False
                    if handling:
                        #add high level event of object type split to log
                        event.append('object type split in activity ' + str(row['ocel:activity']) + ' with object types ' + str(joining_obj_types))
                        event_type.append('s')
                        frequency.append(len(joining_obj_types))
                        #TODO: check times
                        start_time.append(min_time + (curr_seg_count-1)*time_segment)
                        end_time.append(min_time + curr_seg_count*time_segment) 
                        time_frame.append(curr_seg_count)
                        qcid.append(case)
                        related_object.append(joining_obj_types)
                    else:
                        #add high level event of object split to log
                        event.append('object split in activity ' + str(row['ocel:activity']) + ' with objects ' + str(join_df['object'].sum()))
                        event_type.append('s')
                        frequency.append(len(join_df['object'].sum()))
                        #TODO: check times
                        start_time.append(min_time + (curr_seg_count-1)*time_segment)
                        end_time.append(min_time + curr_seg_count*time_segment) 
                        time_frame.append(curr_seg_count)
                        qcid.append(case)
                        related_object.append(join_df['object'].sum())
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
                for i in early_index:
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
                        remaining_obj = objects
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
                    joining_obj_types = join_df['ocel:type'].unique()
                    join_df = join_df.drop_duplicates(subset=['ocel:activity'])
                    #if instance matters for one involved object type, take instances for all object types
                    handling = True
                    for obj in joining_obj_types:
                        if object_handling[obj]==False:
                            handling = False
                    if handling:
                        #add high level event of object type join to log
                        event.append('object type join in activity ' + str(row['ocel:activity']) + ' with objects ' + str(joining_obj_types))
                        event_type.append('j')
                        frequency.append(len(joining_obj_types))
                        #TODO: check times
                        start_time.append(min_time + (curr_seg_count-1)*time_segment)
                        end_time.append(min_time + curr_seg_count*time_segment)  
                        time_frame.append(curr_seg_count) 
                        qcid.append(case)
                        related_object.append(joining_obj_types)    
                    else:
                        #add high level event of object join to log
                        event.append('object join in activity ' + str(row['ocel:activity']) + ' with objects ' + str(join_df['object'].sum()))
                        event_type.append('j')
                        frequency.append(len(join_df['object'].sum()))
                        #TODO: check times
                        start_time.append(min_time + (curr_seg_count-1)*time_segment)
                        end_time.append(min_time + curr_seg_count*time_segment)  
                        time_frame.append(curr_seg_count) 
                        qcid.append(case)
                        related_object.append(join_df['object'].sum())                

    #convert to dataframe
    d = {'event': event, 'event type': event_type, 'frequency': frequency, 'start time': start_time, 'end time': end_time, 'time segment': time_frame, 'quasi case id': qcid, 'related object': related_object}
    hl_log = pd.DataFrame(data=d)
    save_name = './hl_logs/hl_log' + str(curr_seg_count) + '.csv'
    hl_log.to_csv(save_name)
    print(hl_log)

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
    
    for i in range(114):
        str_name = './hl_log_old/hl_log' + str(i) + '.csv'
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

    #filter lagging time events by keeping those above 0.7*max lagging time
    df = copy.copy()
    df_lt = df[df['event type']=='lt']
    df_lt.sort_values('frequency', ascending=False, inplace=True)
    number_lt = len(df_lt)
    filtered_lt = df_lt.head(n=int(number_lt/10))
    #print(filtered_lt)

    #low lagging time not yet implemented

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

    d = {'event1': direct_act1, 'id1': direct_id1, 'time1': direct_time1, 'event2': direct_act2, 'id2': direct_id2, 'time2': direct_time2}
    hl_log = pd.DataFrame(data=d)
    print(hl_log)
    #for i in range(len(not_direct_act1)):
        #text = not_direct_act1[i] + '->' + not_direct_act2[i]
        #print(text)
    print('len of cond: ' + str(len(not_direct_con1)))
    for i in range(len(not_direct_con1)):
        text = '(' + not_direct_con1[i] + '->' + not_direct_con2[i] + ')->' + not_direct_con3[i]
        #print(text)
    #df.to_csv('hl_test.csv')

#main()
pattern_detection()