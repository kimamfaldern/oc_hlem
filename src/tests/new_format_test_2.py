#import pandas as pd
import pm4py
#import os
#from pm4py_core.pm4py.objects.ocel.importer.sqlite.variants import ocel20 as importer

#curr_path = os.getcwd()
#parent_path = os.path.dirname(curr_path)
#root_path = os.path.dirname(parent_path)
dataset = './input_data/order-management2'
#path = root_path + dataset + '.sqlite'
path = dataset + '.sqlite'
#ocel = importer.apply(path)
print(path)
ocel = pm4py.read_ocel2_sqlite(path)
print(ocel)
df = ocel.get_extended_table()
print(df.dtypes)

#ocpn = pm4py.discover_oc_petri_net(ocel)
#pm4py.view_ocpn(ocpn)

objects = pm4py.ocel_get_object_types(ocel)
print(objects)

for obj in objects:
    print(obj)

event_ids = list(df['ocel:eid'])
unique_ids = list(df['ocel:eid'].unique())

print(len(event_ids))
print(len(unique_ids))

