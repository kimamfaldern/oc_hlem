
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa.algo.enhancement.token_replay_based_performance import algorithm as performance_factory
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
#from ocpa.visualization.log.variants.versions import chevron_sequences as cs
import ocpa.objects.log.ocel

filename = "data/datasets/p2p-normal.jsonocel"
ocel = ocel_import_factory.apply(filename)
ocpn = ocpn_discovery_factory.apply(ocel)
diag_params = {'measures': ['act_freq', 'arc_freq', 'object_count', 'pooling_time', 'lagging_time'], 'agg': [
    'mean', 'min', 'max'], 'format': 'svg'}
diag = performance_factory.apply(ocpn, ocel, parameters=diag_params)
print(f'Diagnostics: {diag}')
print('Type of diagnostics: ' + str(type(diag)))
print('First element: ' + str(diag['Receive Goods']))
print('Type of first element: ' + str(type(diag['Receive Goods'])))