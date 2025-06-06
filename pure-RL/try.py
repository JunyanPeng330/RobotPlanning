
import numpy as np
from numpy.linalg import norm
import math

'''
x0 = 3
y0 = 5
z0 = 2
m = np.asarray([x0, y0, z0])
x1 = 4
y1 = 5
z1 = 7
a = np.asarray([x1, y1, z1])
x2 = 8
y2 = 9
z2 = 2
b = np.asarray([x2, y2, z2])
ab=b-a
am=m-a
bm=m-b
r = np.dot(am,ab)/(np.linalg.norm(ab))**2
if r > 0 and r < 1:
	dis = math.sqrt((np.linalg.norm(am))**2 - (r * np.linalg.norm(ab))**2)
elif r >= 1:
	dis = np.linalg.norm(bm)
else:
    dis = np.linalg.norm(am)
#print(dis)
'''


'''
theta = [0, 0, 0, 0, 0, 0]
startTheta = Joint_point(theta)  # joint angels 起始点
T = FK(startTheta)
effector = [T[5][0, 3], T[5][1, 3], T[5][2, 3]]
print(T)
print(effector)
state_1 = [*theta, *effector]
print(state_1)

thetaCurrent = [30, 0, -45, 0, 60, 0]
theta = Joint_point(thetaCurrent)
T = FK(theta)
effectorCurrent = [T[5][0, 3], T[5][1, 3], T[5][2, 3]]
state_1 = [*thetaCurrent, *effectorCurrent]
#print(state_1)

action = [-20, 30, 45, 0, 90, -45]
thetaNext = [thetaCurrent[i]+action[i] for i in range(6)]
print(thetaNext)
theta = Joint_point(thetaNext)
#T = FK(theta)
T = FK(Joint_point(thetaNext))
effectorNext = [T[5][0, 3], T[5][1, 3], T[5][2, 3]]
state_2 = [*thetaNext, *effectorNext]
#print(state_2)


path1 = thetaCurrent.copy()
path1 = path1[np.newaxis,:]
path1 = np.vstack((path1, thetaNext))

print(path1)
'''

'''
currentEffector = [2,4,5]
nextEffector = [3,6,9]
optimalDirection = [0.3, 0.5, 0.2]

realDirection = [nextEffector[i] - currentEffector[i] for i in range(3)]
print(realDirection)
cosine = np.dot(realDirection, optimalDirection)/(norm(realDirection)*norm(optimalDirection))
print(cosine)

‘’‘

g = [0,0,-2]
o = [5,5,0]
e = [6,7,1]

eo = [o[i] - e[i] for i in range(3)]
og = [g[i] - o[i] for i in range(3)]

unit = np.dot(eo, og)/np.dot(eo, eo)
print(unit)
v = [unit * eo[i] for i in range(3)]
print(v)
p = [o[i] + v[i] for i in range(3)]
print('p',p)
perpendicular = [g[i] - p[i] for i in range(3)]
print(perpendicular)
print('oe', eo)
w = 0
for i in range(3):
    w += perpendicular[i] ** 2
v = np.sqrt(w)
avoid = [perpendicular[i] / v for i in range(3)]
print(avoid)
f = [1, 0, 0]
d = [f[i] + avoid[i] for i in range(3)]
print('d', d)
'''

list1 = [[ 1.66766720e+00, 3.94065182e-16,  9.88710680e+00],
 [ 1.66719389e+00,  4.27409502e-02,  9.88673137e+00],
 [ 1.66802395e+00,  7.33675941e-03,  9.88636690e+00],
 [ 1.66779738e+00, 1.49748960e-02,  9.88678802e+00]]

with open("k.txt","a") as ppp:
    ppp.write(''.join(str(list1))+'\n')

list2 = [[ 1.66533041e+00,  1.46916858e-01,  9.88582178e+00],
 [ 1.66734042e+00,  2.85608225e-02,  9.88650560e+00],
 [ 1.66683796e+00,  6.13524424e-02,  9.88676424e+00]]

with open("k.txt","a") as ppp:
    ppp.write(''.join(str(list2))+'\n')



