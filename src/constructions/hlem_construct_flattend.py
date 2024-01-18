import pm4py
#from pm4py.util import constants
#from pm4py.objects.conversion.log import converter as conversion_factory
import pandas as pd
from ocpa.algo.util.process_executions import factory as process_executions_factory
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory

path = "data/datasets/p2p-normal.jsonocel"
ocel = pm4py.read_ocel(path)

objects = ocel.objects
object_types = set(objects['ocel:type'])
print(object_types)

for obj in object_types:
    flattened_log = pm4py.ocel_flattening(ocel, obj)
    path = str(obj) + "_flattened.xes"
    pm4py.write_xes(flattened_log, path)