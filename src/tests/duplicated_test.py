import pm4py
import pandas as pd
import numpy as np
import os

col1 = [1, 2, 3, 4]
col2 = ['A', 'A', 'B', 'C']
d = {'col1': col1, 'col2': col2}

df = pd.DataFrame(data=d)
counts = df.value_counts(subset=['col2'])
test_dict = dict(counts)
test_array =counts.array

duplicates = df.duplicated(subset=['col2'])
print(duplicates)
d = {'dup': duplicates}
df_dup = pd.DataFrame(data=d)
print(df_dup)
df_dup['dup'] = df_dup['dup'].apply(lambda x: x if x else np.nan)
df_dup.dropna(inplace=True)
print(df_dup)
for index, row in df_dup.iterrows():
    print(index)