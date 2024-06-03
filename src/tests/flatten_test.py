import pm4py
import pandas as pd
import os

curr_path = os.getcwd()
parent_path = os.path.dirname(curr_path)
root_path = os.path.dirname(parent_path)
dataset = '/input_data/order-management2'
path = root_path + dataset + '.sqlite'
print(path)
ocel = pm4py.read_ocel2_sqlite(path)

objects = ocel.objects
object_types = set(objects['ocel:type'])
print(object_types)

flattened_log = pm4py.ocel_flattening(ocel, 'employees')

print(flattened_log['case:concept:name'])