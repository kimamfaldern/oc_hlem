import os
import pm4py
import pandas as pd

x = 3
y ='003'

print((x==int(y)))

curr_path = os.getcwd()
parent_path = os.path.dirname(curr_path)
dataset = 'order-management2'
#dataset = 'ocel2-p2p'
path = parent_path + './input_data/' + dataset + '.sqlite'
log = pm4py.read_ocel2_sqlite(path)
print(log)
object_types = ['items', 'orders', 'customers', 'packages', 'products', 'employees']
for t in object_types:
    item_log = pm4py.filter_ocel_object_types(log, [t])
    item_sum = pm4py.ocel_objects_summary(item_log)
    items = item_sum['ocel:oid'].unique()
    print(t + ': ' + str(len(items)))