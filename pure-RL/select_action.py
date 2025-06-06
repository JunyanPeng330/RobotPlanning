import torch
import numpy as np
import random
from DH_parameters import Joint_point
from AAPPFF import APF

def selectAction(ActorList, s):
    actionList = []
    for i in range(len(ActorList)):
        state = s[i]
        state = torch.as_tensor(state, dtype=torch.float, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        a = ActorList[i](state).cpu().detach().numpy()
        actionList.append(a[0])

    return actionList

apf = APF()
def check_collision(apf, path):

    for i in range(path.shape[0]):
        if apf.sphere_collision_check(apf, Joint_point(path[i,:]))[0] == 0:
            return 0
        return 1

def check_path(APF, path):
    summation = 0
    for i in range(path.shape[0] - 1):
        summation += APF.distanceCost(path[i,:], path[i+1,:])
    if check_collision(APF, path) == 1:
        print('safe with obst, traveled length is ', summation)
    else:
        print('collide with obst, traveled length is ', summation)

