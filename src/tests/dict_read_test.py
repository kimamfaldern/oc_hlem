import pickle

with open('result_dict.pkl', 'rb') as fp:
    result = pickle.load(fp)
print('result dictionary')
print(result)