from typing import List
import numpy as np

class robot_DH_param():

    a = [0, 4.185058, 3.918182, 0, 0, 0]
    alpha = [-90, 0, 0, 90, -90, 0]
    d = [1.123748, 0, 0, 0.6676672, 0.6601188, 1]
    theta_o = [-90, -90, 0, 90, 0, 0]

class Point():

    def __init__(self, p: List[float]) -> None:

        self.p = p
        self.x = p[0]
        self.y = p[1]
        self.z = p[2]
        self.vector = np.array(p)  # Vector calculation


class Joint_point():
    """
    Represent a point in joint space, 6 angles theta
    """

    def __init__(self, theta: List[float]) -> None:
        """
        theta : The coordinate of the point in joint space
        """
        self.theta = theta
        self.theta1 = theta[0]
        self.theta2 = theta[1]
        self.theta3 = theta[2]
        self.theta4 = theta[3]
        self.theta5 = theta[4]
        self.theta6 = theta[5]
        self.vector = np.array(theta)

class sphere_obstacle():
    """
    Represent a sphere obstacle
    """
    def __init__(self, center: List[float], r: float) -> None:
        self.x = center[0]
        self.y = center[1]
        self.z = center[2]
        self.r = r

'''
class Cube_obstacle():
    """
    Represent a cube obstacle
    """

    def __init__(self, center: Point, l: float, w: float, h: float) -> None:
        """
        center : The center coordinate of cube obstacle
        l : length          w : width           h : high
        """
        self.x = center.x
        self.y = center.y
        self.z = center.z
        self.l = l
        self.w = w
        self.h = h
'''




