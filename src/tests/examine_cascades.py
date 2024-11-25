from collections import Counter

i = 0
cascade_dict = dict()
while (i<1022):
    i = i+1
    path = './cascade_txts2/cascade' + str(i) + '.txt'
    file = open(path, "r")
    content = file.read()
    file.close()
    content_list = content.split(',')
    content_list.pop()
    cnt = Counter()
    for word in content_list:
        cnt[word] += 1
    cascade_dict[i] = cnt

j = 0
count_dict = dict()
while(j<1022):
    j += 1
    curr_cnt = cascade_dict[j]
    k = j
    count = 0
    while(k<1022):
        k += 1
        if cascade_dict[k]==curr_cnt:
            count += 1
    count_dict[j] = count

sorted_count_dict = sorted(count_dict.items(), key=lambda x:x[1])
print(sorted_count_dict)
#print(count_dict)
most_common_var = max(count_dict, key=count_dict.get)
print(most_common_var)

max_path = './cascade_txts2/cascade' + str(most_common_var) + '.txt'
file = open(path, "r")
content = file.read()
file.close()
print(content)


#print(cascade_dict)
#print(cascade_dict[855]==cascade_dict[860])
#for item in cascade_dict:
#    print(item)
