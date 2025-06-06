import numpy as np
import sys
import array

path = []
file = open("trajectory341.txt", 'r')
row = file.readline()
while row:
    path = np.append(path, row)
    row = file.readline()
file.close()

#print(path)
#print(type(path), np.size(path))

#l0 = list(path)
#print(path[2])

final_path = str('{')
for i in range(len(path)):
    #print(i, path[i], len(path[i]))
    each_row = path[i].split()
    #print(i, each_row)
    #print(each_row[1])

    if i < len(path):  #每行
        new_row_start = str('{')
        each_row[1] = str(each_row[1]) + 'f' + ','
        each_row[2] = str(-1*float(each_row[2])) + 'f' + ','
        each_row[3] = str(-1*float(each_row[3])) + 'f' + ','
        each_row[4] = str(each_row[4]) + 'f' + ','
        each_row[5] = str(each_row[5]) + 'f' + ','
        each_row[6] = str(0) + 'f'
        new_row_end = str('}, ')

        new_row = new_row_start + each_row[1] + each_row[2] + each_row[3] + each_row[4] + each_row[5] + each_row[6] + new_row_end
        print(new_row)

        final_path = final_path + new_row

print('final', final_path)









