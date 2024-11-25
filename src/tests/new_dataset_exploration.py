import pm4py
import pandas as pd
import os

def main():    
    #define data set and paths
    curr_path = os.getcwd()
    parent_path = os.path.dirname(curr_path)
    #dataset = 'order-management2'
    dataset = 'ocel2-p2p'
    path = parent_path + './input_data/' + dataset + '.sqlite'
    print(path)
    #read log and dataframe
    log = pm4py.read_ocel2_sqlite(path)
    dfg = pm4py.algo.discovery.ocel.ocdfg.variants.classic.apply(log)
    vis_dfg = pm4py.vis.view_ocdfg(dfg)
    print(vis_dfg)
    df = log.get_extended_table()
    df.sort_values(by='ocel:timestamp', inplace=True)
    object_list = ['goods receipt:1287', 'invoice receipt:838', 'material:43', 'payment:6', 'purchase_order:361', 'purchase_requisition:30:pr_trigger_30', 'quotation:65']
    object_type_list = ['ocel:type:goods receipt', 'ocel:type:invoice receipt', 'ocel:type:material', 'ocel:type:payment', 'ocel:type:purchase_order', 'ocel:type:purchase_requisition', 'ocel:type:quotation']
    count = 0
    for o in object_list:
        new_df = df.copy()
        new_df = new_df.dropna(subset=[object_type_list[count]])
        new_df[object_type_list[count]] = new_df[object_type_list[count]].apply(lambda x: True if o in x else False)
        new_df = new_df[new_df[object_type_list[count]]==True]
        print(new_df['ocel:activity'])
        count = count + 1

main()