import sys
print(sys.getrecursionlimit())

def recursive_function(element, list):
    print(str(list) + str(element))
    new_list = list.copy()
    new_list.append(element)
    if element<10:
        recursive_function(element+1, new_list)
    if element<500:
        recursive_function(element+100, new_list)

recursive_function(1, [])