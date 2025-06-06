from TD import AgentTD3, ReplayBuffer
from AAPPFF import APF
from setup_seed import setup_seed
from transAction import transformAction
import numpy as np
import time
from obstacle import Joint_point, environment
import random
import matplotlib.pyplot as plt


import random
import torch

if __name__ == '__main__':
    setup_seed(1)   # 设置随机数种子

    apf = APF()
    obs_dim = 3 * apf.numberOfSphere
    act_dim = 6
    act_bound = [-1, 1]

    agent = AgentTD3()
    agent.init(256,obs_dim,act_dim)
    buffer = ReplayBuffer(int(1e5), obs_dim, act_dim, False, True)

    gamma = 0.98
    MAX_EPISODE = 700
    MAX_STEP = 200
    batch_size = 256
    update_cnt = 0
    rewardList = []
    maxReward = -np.inf
    update_every = 50

    best_path_joints = []
    best_path_effector = []

    loss_q_List = []
    loss_pi_List = []

    best_path_steps = []
    best_path_episode = []
    feasible_path_steps = []
    feasible_path_episode = []

    tot_time = 0
    start_training_time = time.time()

    for episode in range(MAX_EPISODE):
        start = time.time()
        print('---------------------------------------------------------------episode=', episode)

        stateCurrent = apf.starting  # original start thetas
        thetaCurrent = apf.startTheta
        effectorCurrent = apf.startEffector

        apf.reset()  # reset
        rewardSum = 0
        stateBefore = [None, None, None, None, None, None, None, None, None]  # first 6 : joints; last 3 : effector coordinate
        thetaBefore = [None, None, None, None, None, None]
        effectorBefore = [None, None, None]

        path1 = apf.startTheta.copy()
        path2 = apf.startEffector.copy()

        collision = 0
        step_count = 0

        loss_pi_sum = 0
        loss_q_sum = 0



        for j in range(MAX_STEP):
            step_count += 1

            obsDicq, _ = apf.calculateDynamicState(effectorCurrent)
            obs_sphere = obsDicq['sphere']  # , obsDicq['cylinder'], obsDicq['cone'] #, obs_cylinder, obs_cone
            obs_mix = obs_sphere  # + obs_cylinder + obs_cone
            obs = np.array([])

            for k in range(len(obs_mix)):
                obs = np.hstack((obs, obs_mix[k])) # 拼接状态为一个1*n向量

            if episode > 50:
                action = agent.select_action(obs)
                action = transformAction(action,act_bound,act_dim)
                # 分解动作向量
                action = action[0:apf.numberOfSphere]

            else:
                action = [random.uniform(act_bound[0],act_bound[1]) for k in range(6)]
            #print('action:',action)

            # 与环境交互
            stateNext, thetaNext, effectorNext = apf.getNextState(action, thetaCurrent)
            path1 = np.row_stack((path1, thetaNext))
            path2 = np.row_stack((path2, effectorNext))
            obsDicqNext, nearestObs = apf.calculateDynamicState(effectorNext)
            obs_sphere_next = obsDicqNext['sphere']
            # obs_mix_next = obs_sphere_next + obs_cylinder_next + obs_cone_next
            obs_next = np.array([])

            for k in range(len(obs_sphere_next)):
                obs_next = np.hstack((obs_next, obs_sphere_next[k]))

            flag = apf.sphere_collision_check(Joint_point(thetaNext))

            if flag[0] == 0:
                collision += 1
            optDirection = apf.getUnitCompositeForce(effector=effectorCurrent, eta1List=apf.eta0, epsilon=apf.epsilon0, nearestObstacle=nearestObs)
            #print('optDirection:',optDirection)

            reward = apf.getRewardA(apf, flag, effectorNext, effectorCurrent, optDirection)
            rewardSum += reward
            #print('reward:',reward)

            done = True if apf.distanceCost(apf.goal, effectorNext) < apf.threshold else False
            #agent.Replay_Buffer.store(obs, action, reward, obs_next, done)
            mask = 0.0 if done else gamma
            other = (reward,mask,*action)
            #print('obs:', obs)
            #print('other:', other)
            buffer.append_buffer(obs,other)


            if episode >= 50 and j % update_every == 0:
                loss_q, loss_pi = agent.update_net(buffer,update_every,batch_size,1)
                update_cnt += update_every

                loss_pi_sum += loss_pi
                loss_q_sum += loss_q

                # record the loss, loss_q, loss_pi
                loss_pi_List.append(loss_pi)
                if episode > 51:
                    loss_q_List.append(loss_q)


            if done:
                if collision == 0:
                    feasible_path_steps.append(step_count)
                    feasible_path_episode.append(episode)
                    with open("figure/feasible_path.txt", "a") as feasible_path:
                        feasible_path.write(''.join(str(path1)) + '\n')

                break


            stateBefore = stateCurrent
            stateCurrent = stateNext
            thetaBefore = thetaCurrent
            thetaCurrent = thetaNext
            effectorBefore = effectorCurrent
            effectorCurrent = effectorNext

        print('Episode:', episode, 'collision:', collision, 'Reward:%f' % rewardSum, 'noise:%f' % agent.policy_noise, 'steps:%f' % step_count)
        rewardList.append(round(rewardSum, 2))

        loss_pi_List.append(loss_pi_sum*0.005)
        loss_q_List.append(loss_q_sum*0.005)


        if rewardSum > maxReward and collision == 0:
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Reward is the best so far! Model is saved!')
            maxReward = rewardSum
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& rewardSum', rewardSum)
            print('new best path1:', path1)
            print('new best path2:', path2)
            print('new best goaling:', effectorNext)
            torch.save(agent.act, 'trained_models/323.pkl')

            best_path_joints = path1
            best_path_effector = path2
            with open("figure/best_path.txt", "a") as best_path:
                best_path.write(''.join(str(path1)) + '\n')

            # record th number of steps of best path
            best_path_steps.append(step_count)
            best_path_episode.append(episode)

        time_one_episode = time.time() - start
        print('time one episode: ', time_one_episode)

    time_tot = time.time() - start_training_time + tot_time
    print('time total: ', time_tot)
    #print('best path for joints:', best_path_joints)
    #print('best path for effector:', best_path_effector)

    episodeList = list(range(len(rewardList)))
    fig_path = 'figure/' + 'reward' + '.png'
    plt.plot(episodeList, rewardList)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('reward')
    # plt.show()
    plt.savefig(fig_path)
    plt.close()

    idx_loss_q = [i for i in range(1, len(loss_q_List) + 1)]
    fig_path = 'figure/' + 'loss_q' + '.png'
    plt.plot(idx_loss_q, loss_q_List)
    plt.xlabel('episode')
    plt.ylabel('loss Q')
    plt.title('loss of Q network')
    # plt.show()
    plt.savefig(fig_path)
    plt.close()

    idx_loss_pi = [i for i in range(1, len(loss_pi_List) + 1)]
    fig_path = 'figure/' + 'loss_pi' + '.png'
    plt.plot(idx_loss_pi, loss_pi_List)
    plt.xlabel('episode')
    plt.ylabel('loss Policy')
    plt.title('loss of Policy network')
    # plt.show()
    plt.savefig(fig_path)
    plt.close()

    idx_best_path = [i for i in range(1, len(best_path_steps) + 1)]
    fig_path = 'figure/' + 'best_path_steps_(a)' + '.png'
    plt.plot(idx_best_path, best_path_steps)
    plt.ylabel('best path steps')
    plt.title('number of steps of best paths')
    # plt.show()
    plt.savefig(fig_path)
    plt.close()

    fig_path = 'figure/' + 'best_path_steps_(b)' + '.png'
    plt.plot(best_path_episode, best_path_steps)
    plt.xlabel('episode')
    plt.ylabel('best path steps')
    plt.title('number of steps of best paths')
    # plt.show()
    plt.savefig(fig_path)
    plt.close()

    idx_feasible_path = [i for i in range(1, len(feasible_path_steps) + 1)]
    fig_path = 'figure/' + 'feasible_path_steps_(a)' + '.png'
    plt.plot(idx_feasible_path, feasible_path_steps)
    plt.ylabel('feasible path steps')
    plt.title('number of steps of feasible paths')
    # plt.show()
    plt.savefig(fig_path)
    plt.close()

    fig_path = 'figure/' + 'feasible_path_steps_(b)' + '.png'
    plt.plot(feasible_path_episode, feasible_path_steps)
    plt.xlabel('episode')
    plt.ylabel('feasible path steps')
    plt.title('number of steps of feasible paths')
    # plt.show()
    plt.savefig(fig_path)
    plt.close()


    plotfile = open("figure/plot.txt", "w")
    plotfile.writelines(str(time_tot))
    plotfile.write('\n')
    plotfile.writelines(str(best_path_episode))
    plotfile.write('\n')
    plotfile.writelines(str(best_path_steps))
    plotfile.write('\n')
    plotfile.writelines(str(feasible_path_episode))
    plotfile.write('\n')
    plotfile.writelines(str(feasible_path_steps))
    plotfile.write('\n')
    plotfile.writelines(str(episodeList))
    plotfile.write('\n')
    plotfile.writelines(str(rewardList))









