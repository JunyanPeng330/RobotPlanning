import numpy as np
from forward_kinematics import FK
from DH_parameters import Point, Joint_point, sphere_obstacle

class obstacle_env:
    def __init__(self):
        '''
        self.Cobstacle = np.array([[3, 2, 2],
                                   [3, 4, 2],
                                   [1, 2, 9],
                                   [1, 0, 8],
                                   [3, 3, 7],
                                   [5, 4, 4],
                                   [2, 3, 3],
                                   [6, 5, 1]
                                   ], dtype=float)  # sphere Obstacles coordinates
        self.Robstacle = np.array([0.3, 0.3, 0.4, 0.3, 0.5, 0.5, 0.4, 0.4], dtype=float)  # their radius
        '''
        self.Cobstacle = np.array([[3, 2, 2],
                                   [3, 4, 2],
                                   [1, 2, 9],
                                   [1, 0, 8],
                                   [3, 3, 7],
                                   [5, 4, 4],
                                   [2, 3, 3]
                                   ], dtype=float)  # sphere Obstacles coordinates
        self.Robstacle = np.array([0.3, 0.3, 0.4, 0.3, 0.3, 0.3, 0.4], dtype=float)  # their radius

        self.goal = [4, 3, 4]


        self.startTheta = [0, 0, 0, 0, 0, 0]  # joint angels starting
        T = FK(Joint_point(self.startTheta))
        self.startEffector = [T[5][0, 3], T[5][1, 3], T[5][2, 3]]
        startingState = [*self.startTheta, *self.startEffector]
        self.starting = startingState

environment = {"env_sphere": obstacle_env()}


