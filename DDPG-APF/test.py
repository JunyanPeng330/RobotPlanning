import torch
import numpy as np
from AAPPFF import APF
from get_distance import distance_point_to_point
from DH_parameters import Joint_point, Point



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    apf = APF()  # 实例化APF问题
    Actor = torch.load('trained_models/testModel3.pkl', map_location=device) #加载模型
    actionList = np.array([]) #, 'cylinder':np.array([]), 'cone':np.array([])

    stateCurrent = apf.starting  # original start thetas
    thetaCurrent = apf.startTheta
    effectorCurrent = apf.startEffector
    print('stateCurrent-', stateCurrent)
    print('thetaCurrent-', thetaCurrent)
    print('effectorCurrent-', effectorCurrent)

    stateBefore = [None, None, None, None, None, None, None, None, None]  # first 6 : joints; last 3 : effector coordinate
    thetaBefore = [None, None, None, None, None, None]
    effectorBefore = [None, None, None]
    print('stateBefore-', stateBefore)
    print('thetaBefore-', thetaBefore)
    print('effectorBefore-', effectorBefore)

    path1 = apf.startTheta.copy()
    path2 = apf.startEffector.copy()

    collision = 0

    rewardSum = 0
    stepCount = 0
    effector_path_length = 0
    for i in range(200):
        stepCount += 1
        obsDicq, _ = apf.calculateDynamicState(effectorCurrent)
        obs_sphere = obsDicq['sphere']
        obs_mix = obs_sphere  # + obs_cube
        obs = np.array([])
        for k in range(len(obs_mix)):
            obs = np.hstack((obs, obs_mix[k]))
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        action = Actor(obs).cpu().detach().numpy()

        print("action:", action)

        # action decomposing [sphereObst, cubeObst]
        action_sphere = action[0:apf.numberOfSphere]
        actionList = np.append(actionList, action_sphere)

        stateNext, thetaNext, effectorNext = apf.getNextState(action, thetaCurrent)

        flag = apf.sphere_collision_check(Joint_point(thetaNext))
        print('flag:', flag)
        if flag[0] == 0:
            collision += 1

        obsDicqNext, nearestObs = apf.calculateDynamicState(effectorNext)
        obs_sphere_next = obsDicqNext['sphere']
        obs_next = np.array([])
        for k in range(len(obs_sphere_next)):
            obs_next = np.hstack((obs_next, obs_sphere_next[k]))
        #print('obs_next:', obs_next)

        optDirection = apf.getUnitCompositeForce(effectorCurrent, eta1List=apf.eta0, epsilon=apf.epsilon0, nearestObstacle=nearestObs)
        #print('optDirection:', optDirection)
        reward = apf.getRewardA(apf, flag, effectorNext, effectorCurrent, optDirection)
        #print('rewardA:', rewardA)
        #rewardB = apf.getRewardB(effectorCurrent, effectorNext, optDirection, nearestObs)
        #print('rewardB:', rewardB)
        #reward = rewardA + rewardB
        print('reward:', reward)
        rewardSum += reward
        print('rewardSum:', rewardSum)

        effector_path_length += distance_point_to_point(effectorNext, effectorCurrent)

        stateBefore = stateCurrent
        stateCurrent = stateNext
        thetaBefore = thetaCurrent
        thetaCurrent = thetaNext
        effectorBefore = effectorCurrent
        effectorCurrent = effectorNext
        #print('stateBefore--', stateBefore)
        #print('stateCurrent--', stateCurrent)
        #print('thetaBefore--', thetaBefore)
        #print('thetaCurrent--', thetaCurrent)
        #print('effectorBefore--', effectorBefore)
        print('effectorCurrent--', effectorCurrent)


        path1 = np.row_stack((path1, thetaNext))
        #print('path of joint:', path1)
        path2 = np.row_stack((path2, effectorNext))
        #print('path of effector:', path2)

        if apf.distanceCost(apf.goal, effectorNext) < apf.threshold:
            break

    #check_path(APF, path1)

    print('this path~ rewardSum:', rewardSum)
    print('this path~ stepCount:', stepCount)
    print("effector's path length:", effector_path_length)
    print('collision detection: ', collision)
    print('path1:', path1)
    print('path2:', path2)

