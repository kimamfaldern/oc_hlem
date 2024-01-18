import pm4py
#from pm4py.util import constants
#from pm4py.objects.conversion.log import converter as conversion_factory
import pandas as pd
#from ocpa_test.ocpa.algo.util.process_executions import factory as process_executions_factory
#from ocpa_test.ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa_test.ocpa.algo.util.process_executions import factory as process_executions_factory
from ocpa_test.ocpa.objects.log.importer.ocel2.sqlite import factory as ocel_import_factory
from ocpa_test.ocpa.objects.log.importer.ocel import factory as ocel_import_factory_old

path = "order-management2.sqlite"
ocel_pm4py = pm4py.read_ocel2_sqlite(path)
filtered_ocel = pm4py.filter_ocel_object_attribute(ocel_pm4py, "ocel:type", ["orders", "items", 'packages'], positive=True)
print(filtered_ocel.get_extended_table())
objects = ['orders', 'items', 'packages']
print('done with pm4py input')
save_path = "./order-management-test.sqlite"
pm4py.write_ocel(filtered_ocel, save_path)
print('done saving')
ocel_ocpa = ocel_import_factory.apply(path)
print('done with ocpa input')
print(ocel_ocpa.object_types)
print('done with test')
objects = pm4py.ocel_get_object_types(filtered_ocel)
print(objects)
extended_df = filtered_ocel.get_extended_table()
print('done with objects and table')

#process executions
exec = process_executions_factory.apply(ocel_ocpa, variant='connected_components')#, parameters={"execution_extraction": "leading_type", "leading_type": "employees",})
print('done with variants')
#print(exec)
i=0
cases = dict()
eventids = exec[0]
#first execution
#print('exec[0]:')
#print(exec[0])
print('number of executions:')
print(len(eventids))
for execution in eventids:
    print('len of execution:')
    print(len(execution))
    for x in execution:
        #print(x)
        #print(type(x))
        #print(extended_df.loc[x, 'ocel:eid'])
        #print(type(extended_df.loc[x, 'ocel:eid']))
        #event_name = extended_df.loc[x, 'ocel:eid']
        #print(event_name)
        #cases[str(event_name)] = str(i)
        cases[str(x)] = str(i)
    i=i+1
#df = ocel_pm4py.events
#df.insert(0, "id", 0, True)
#df['id'] = df['ocel:eid'].map(cases)

print('extended_df vorher:')
print(extended_df)
new_df = extended_df.copy()
new_df['ocel:process:execusion'] = extended_df['ocel:eid']
new_df['ocel:process:execusion'].replace(cases, inplace=True)
print('new df nachher:')
print(new_df)

print('execusions:' + new_df['ocel:process:execusion'].unique())
#event_log = pm4py.convert_to_event_log(new_df)
#print(event_log)
#path = "./output2.xes"
#pm4py.write_xes(event_log, path)