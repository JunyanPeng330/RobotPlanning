import numpy as np
from obstacle import environment
from DH_parameters import Point, Joint_point
from forward_kinematics import FK
from get_distance import distance_line_to_point
from numpy.linalg import norm

class APF:
    def __init__(self):

        env = 'env_sphere'

        self.Cobstacle = environment[env].Cobstacle        # 球障碍物坐标
        self.Robstacle = environment[env].Robstacle  # 球半径
        self.numberOfSphere = environment[env].Cobstacle.shape[0]       # 球形障碍物的数量

        self.goal = environment[env].goal   # 目标点
        self.starting = environment[env].starting      # starting state
        self.startTheta = environment[env].startTheta
        self.startEffector = environment[env].startEffector

        self.dgoal = 10                                    # 当q与goal距离超过它时将衰减一部分引力
        self.r0 = 2                                       # 斥力超过这个范围后将不存在
        self.threshold = 0.1                              # q与goal距离小于它时终止训练

        #-------------一些参考参数可选择使用-------------#
        self.epsilon0 = 0.8
        self.eta0 = [0.3, 0.2, 0.3, 0.3, 0.3, 0.4, 0.2]  # [0.3, 0.2, 0.3, 0.3, 0.3, 0.4, 0.2, 0.2]  #


    def reset(self):  # reset the environment
        self.path1 = self.startTheta.copy()
        #self.path1 = self.path1[np.newaxis, :]
        self.path2 = self.startEffector.copy()
        #self.path2 = self.path2[np.newaxis, :]  # 增加一个维度

    def calculateDynamicState(self, effector):  # 计算空间一个坐标下指向每个障碍物中心的向量,返回字典，key为障碍物名称; nearest obstacle
        dicObstacles = {'sphere':[]}   #
        minDistance = APF.distanceCost(effector, self.Cobstacle[0,:])  # the first obstacle
        nearestObstacle = self.Cobstacle[0,:]
        for i in range(self.numberOfSphere):
            dicObstacles['sphere'].append(self.Cobstacle[i,:] - effector)
            distance = APF.distanceCost(self.Cobstacle[i,:], effector)
            if distance < minDistance:
                minDistance = distance
                nearestObstacle = self.Cobstacle[i,:]
        #dicObst = dict(dicObstacles)

        return dicObstacles, nearestObstacle

    def inRepulsionArea(self, effector):  # 计算一个点位r0半径范围内的障碍物索引, 返回字典{'sphere':[1,2,..],'cylinder':[0,1,..]}  2021.1.6
        dic = {'sphere':[]} #, 'cylinder':[], 'cone':[]
        for i in range(self.numberOfSphere):
            if self.distanceCost(effector, self.Cobstacle[i, :]) < self.r0:
                dic['sphere'].append(i)

        return dic


    def attraction(self, effector, epsilon):  # 计算引力的函数
        r = self.distanceCost(effector, self.goal)
        if r <= self.dgoal:
            fx = epsilon * (self.goal[0] - effector[0])
            fy = epsilon * (self.goal[1] - effector[1])
            fz = epsilon * (self.goal[2] - effector[2])
        else:
            fx = self.dgoal * epsilon * (self.goal[0] - effector[0]) / r
            fy = self.dgoal * epsilon * (self.goal[1] - effector[1]) / r
            fz = self.dgoal * epsilon * (self.goal[2] - effector[2]) / r
        return np.array([fx, fy, fz])

    def repulsion(self, effector, eta):  # 斥力计算函数
        f0 = np.array([0, 0, 0])  # 初始化斥力的合力
        Rq2qgoal = self.distanceCost(effector, self.goal)
        for i in range(self.Cobstacle.shape[0]):       #球的斥力
            r = self.distanceCost(effector, self.Cobstacle[i, :])
            if r <= self.r0:
                tempfvec = eta * (1 / r - 1 / self.r0) * Rq2qgoal ** 2 / r ** 2 * self.differential(effector, self.Cobstacle[i, :]) \
                           + eta * (1 / r - 1 / self.r0) ** 2 * Rq2qgoal * self.differential(effector, self.goal)
                f0 = f0 + tempfvec
            else:
                tempfvec = np.array([0, 0, 0])
                f0 = f0 + tempfvec
        return f0

    def repulsionForOneObstacle(self, effector, eta, qobs): #这个版本的斥力计算函数计算的是一个障碍物的斥力
        f0 = np.array([0, 0, 0])  # 初始化斥力的合力
        Rq2qgoal = self.distanceCost(effector, self.goal)
        r = self.distanceCost(effector, qobs)
        if r <= self.r0:
            tempfvec = eta * (1 / r - 1 / self.r0) * Rq2qgoal ** 2 / r ** 2 * self.differential(effector, qobs) \
                       + eta * (1 / r - 1 / self.r0) ** 2 * Rq2qgoal * self.differential(effector, self.goal)
            f0 = f0 + tempfvec
        else:
            tempfvec = np.array([0, 0, 0])
            f0 = f0 + tempfvec
        return f0

    def differential(self, effector, other):   #向量微分
        output1 = (effector[0] - other[0]) / self.distanceCost(effector, other)
        output2 = (effector[1] - other[1]) / self.distanceCost(effector, other)
        output3 = (effector[2] - other[2]) / self.distanceCost(effector, other)
        return np.array([output1, output2, output3])

    def getUnitCompositeForce(self, effector, eta1List, epsilon, nearestObstacle): #, eta2List, eta3List
        Attraction = self.attraction(effector, epsilon)  # 计算引力
        Repulsion = np.array([0,0,0])
        for i in range(len(eta1List)): #对每个球形障碍物分别计算斥力并相加
            tempD = self.distanceCost(effector, self.Cobstacle[i,:])
            repPoint = effector + (self.Cobstacle[i,:] - effector) * (tempD - self.Robstacle[i]) / tempD
            Repulsion = Repulsion + self.repulsionForOneObstacle(effector, eta1List[i], repPoint)

        compositeForce = Attraction + Repulsion  # 合力 = 引力 + 斥力
        forceDirection = self.getUnitVec(compositeForce)  # 力单位化，apf中力只用来指示移动方向(length of vector = 1)


        disOE = APF.distanceCost(effector, nearestObstacle)
        disGE = APF.distanceCost(effector, self.goal)
        OG = [self.goal[i] - nearestObstacle[i] for i in range(3)]
        EO = [nearestObstacle[i] - effector[i] for i in range(3)]
        OE = [effector[i] - nearestObstacle[i] for i in range(3)]
        cosGOE = np.dot(OG, OE) / (norm(OG) * norm(OE))

        unit = np.dot(EO, OG)/np.dot(EO, EO)
        v = [unit * EO[i] for i in range(3)]
        p = [nearestObstacle[i] + v[i] for i in range(3)]
        perpendicular = [self.goal[i] - p[i] for i in range(3)]
        w = 0
        for i in range(3):
            w += perpendicular[i] ** 2
        u = np.sqrt(w)
        avoid = [perpendicular[i] / u for i in range(3)]  # UNIT-DIRECTION TO AVOID LOCAL OPTIMAL

        if disOE/disGE < 0.5 and cosGOE < -0.85:
            eltraDirection = [forceDirection[i] + avoid[i] for i in range(3)]

            return eltraDirection
        else:
            return forceDirection


    def getRewardA(self, apf, flag, nextEffector, currentEffector, optimalDirection):

        disGE = APF.distanceCost(nextEffector, apf.goal)

        realDirection = [nextEffector[i] - currentEffector[i] for i in range(3)]
        cosSigma = np.dot(realDirection, optimalDirection) / (norm(realDirection) * norm(optimalDirection))
        # hope cosSigma is bigger

        if flag[0] == 0:  # collides

            #reward = -0.1 * disGE
            rewardF = 2*cosSigma - 2*disGE - 2

        else:  # not collides
            rewardF = 2*cosSigma - disGE - 2

        return rewardF


    def sphere_collision_check(self, joint_point: Joint_point) -> bool:  # obstacles: all spheres in the env

        [T2, T3, T4, T5, T6, T7] = FK(joint_point)

        # check link1:
        o1 = Point([0, 0, 0])
        o2 = Point([T2[0, 3], T2[1, 3], T2[2, 3]])
        # print('link 1:', T2[0, 3], T2[1, 3], T2[2, 3])
        for i in range(self.numberOfSphere):
            if distance_line_to_point(o1, o2, self.Cobstacle[i, :]) <= self.Robstacle[i]:
                link1_collision = True
                return np.array([0, 1, i])
            else:
                link1_collision = False

        # check link2:
        p0 = Point([T2[0, 3], T2[1, 3], T2[2, 3]])
        # print('link 2   T2:', T2[0,3], T2[1,3], T2[2,3])
        p1 = Point([T3[0, 3], T3[1, 3], T3[2, 3]])
        for i in range(self.numberOfSphere):
            if distance_line_to_point(p0, p1, self.Cobstacle[i, :]) <= self.Robstacle[i]:
                link2_collision = True
                return np.array([0, 2, i])
            else:
                link2_collision = False

        # check link3:
        p0 = Point([T3[0, 3], T3[1, 3], T3[2, 3]])
        # print('link 3   T3:', T3[0, 3], T3[1, 3], T3[2, 3])
        p1 = Point([T4[0, 3], T4[1, 3], T4[2, 3]])
        # print('link 3   T4:', T4[0, 3], T4[1, 3], T4[2, 3])
        for i in range(self.numberOfSphere):
            if distance_line_to_point(p0, p1, self.Cobstacle[i, :]) <= self.Robstacle[i]:
                link3_collision = True
                return np.array([0, 3, i])
            else:
                link3_collision = False

        # check link4:
        p0 = Point([T4[0, 3], T4[1, 3], T4[2, 3]])
        # print('link 4   T4:', T4[0, 3], T4[1, 3], T4[2, 3])
        p1 = Point([T5[0, 3], T5[1, 3], T5[2, 3]])
        # print('link 4   T5:', T5[0, 3], T5[1, 3], T5[2, 3])
        for i in range(self.numberOfSphere):
            if distance_line_to_point(p0, p1, self.Cobstacle[i, :]) <= self.Robstacle[i]:
                link4_collision = True
                return np.array([0, 4, i])
            else:
                link4_collision = False

        # check link5:
        p0 = Point([T5[0, 3], T5[1, 3], T5[2, 3]])
        p1 = Point([T6[0, 3], T6[1, 3], T6[2, 3]])
        # print('link 4   T5:', T5[0, 3], T5[1, 3], T5[2, 3])
        for i in range(self.numberOfSphere):
            if distance_line_to_point(p0, p1, self.Cobstacle[i, :]) <= self.Robstacle[i]:
                link5_collision = True
                return np.array([0, 5, i])
            else:
                link5_collision = False

        return np.asarray([1, -1, -1])  # no collision


    def getNextState(self, action, currentTheta):   #
        """
        当qBefore为[None, None, None]时，意味着q是航迹的起始点，下一位置不需要做运动学约束，否则进行运动学约束
        """
        thetaNext = [currentTheta[i] + action[i] for i in range(6)]
        theta = Joint_point(thetaNext)
        T = FK(theta)
        effectorNext = [T[5][0, 3], T[5][1, 3], T[5][2, 3]]
        stateNext = [*thetaNext, *effectorNext]
        #_, _, _, _, effectorNext = self.kinematicConstrant(effector, effectorBefore, effectorNext)

        return stateNext, thetaNext, effectorNext


    @staticmethod
    def distanceCost(point1, point2):  # 求两点之间的距离函数
        p1 = np.asarray(point1)
        p2 = np.asarray(point2)
        return np.sqrt(np.sum((p1 - p2) ** 2))

    @staticmethod
    def angleVec(vec1, vec2):  # 计算两个向量之间的夹角
        temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
        temp = np.clip(temp,-1,1)  # 可能存在精度误差导致上一步的temp略大于1，因此clip
        theta = np.arccos(temp)
        return theta

    @staticmethod
    def getUnitVec(vec):   #单位化向量方法
        unitVec = vec / np.sqrt(np.sum(vec ** 2))
        return unitVec

    def calculateLength(self):

        sum = 0  # 轨迹距离初始化
        for i in range(self.path2.shape[0] - 1):
            sum += apf.distanceCost(self.path2[i, :], self.path2[i + 1, :])
        return sum



if __name__ == "__main__":
    apf = APF()
    apf.loop()
    #apf.saveCSV()
    #print('轨迹距离为：',apf.calculateLength())





