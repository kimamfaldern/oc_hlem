import pandas as pd
import pm4py

ocel = pm4py.read_ocel2_sqlite('order-management.sqlite')
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

