import numpy as np
from math import sin, cos, radians

def rotx(angle):
    radian_angle = radians(angle)
    T = np.matrix([[1,0,0,0],[0,cos(radian_angle),-1*sin(radian_angle),0],[0,sin(radian_angle),cos(radian_angle),0],[0,0,0,1]])
    return T

def roty(angle):
    radian_angle = radians(angle)
    T = np.matrix([[cos(radian_angle),0,sin(radian_angle),0],[0,1,0,0],[-1*sin(radian_angle),0,cos(radian_angle),0],[0,0,0,1]])
    return T

def rotz(angle):
    radian_angle = radians(angle)
    T = np.matrix([[cos(radian_angle),-1*sin(radian_angle),0,0],[sin(radian_angle),cos(radian_angle),0,0],[0,0,1,0],[0,0,0,1]])
    return T

def trans(x,y,z):
    T=np.matrix([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])
    return T