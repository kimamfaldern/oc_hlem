import pm4py

log1 = pm4py.read_xes('output_safe.xes')
df1 = pm4py.convert_to_dataframe(log1)
df1.sort_values(by=['time:timestamp'], inplace=True)
print(df1)

log2 = pm4py.read_xes('output.xes')
df2 = pm4py.convert_to_dataframe(log2)
df2.sort_values(by=['time:timestamp'], inplace=True)
print(df2)
