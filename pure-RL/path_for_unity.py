import numpy as np
import sys
import array

path = []
file = open(r"trajectory1.txt")
row = file.readline()
while row:
    num = str(row)
    print(num)
    print(type(num), np.size(num))
    num.split()

    print(num)

    path = np.append(path, num)
    row = file.readline()

file.close()
#print(path)
#print(type(path), np.size(path))

#l0 = list(path)
#print(path[2][3])
#rotation = [[] for i in range(len(path))]
rotation = str('{')
for i in range(len(path)):

    if i < len(path)-1:
        rot = str('{')
        for j in range(len(path[i])-1):  # first 5, add a 'f,'
            if j == 0 or j == 4 or j == 5:  # the 1st and 5th, stay the same
                path[i][j] = str(path[i][j]) + 'f'
                rot = rot + path[i][j] + ','
            else:  #the 2nd. 3rd, 4th, *(-1)
                path[i][j] = -1 * path[i][j]
                path[i][j] = str(path[i][j]) + 'f'
                rot = rot + path[i][j] + ','


        rot = rot + path[i][5] + '}'

        rotation = rotation + rot +','

    if i == len(path) - 1:  # the last row of path,  end with {}
        rot = str('{')
        for j in range(len(path[i]) - 1):  # first 5, add a 'f,'
            if j == 0 or j == 4 or j == 5:  # the 1st and 5th, stay the same
                path[i][j] = str(path[i][j]) + 'f'
                rot = rot + path[i][j] + ','
            else:  # the 2nd. 3rd, 4th, *(-1)
                path[i][j] = -1 * path[i][j]
                path[i][j] = str(path[i][j]) + 'f'
                rot = rot + path[i][j] + ','


        rot = rot + path[i][5] + '}'

        rotation = rotation + rot +'}'

print(len(path))

print(rotation)









