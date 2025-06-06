from DDPG import DDPG
from AAPPFF import APF
from setup_seed import setup_seed
import random
import numpy as np
import torch
from obstacle import Joint_point, environment
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
    tot_time = 0
    start_training_time = time.time()

    setup_seed(11)

    MAX_EPISODE = 700
    MAX_STEP = 200
    update_every = 50
    batch_size = 256
    noise = 0.06
    update_cnt = 0
    rewardList = []
    maxReward = -np.inf


    apf = APF()
    obs_dim = 3 * apf.numberOfSphere  # obstacles' coordinates relative to the effector
    act_dim = 6   # 6 joints
    act_bound = [-1, 1]

    agent = DDPG(obs_dim, act_dim, act_bound, noise)

    best_path_joints = []
    best_path_effector = []

    loss_q_List = []
    loss_pi_List = []


    best_path_steps = []
    best_path_episode = []
    feasible_path_steps = []
    feasible_path_episode = []


    for episode in range(MAX_EPISODE):

        start = time.time()
        print('---------------------------------------------------------------episode=', episode)

        stateCurrent = apf.starting  # original start thetas
        thetaCurrent = apf.startTheta
        effectorCurrent = apf.startEffector
        #print('stateCurrent-', stateCurrent)
        #print('thetaCurrent-', thetaCurrent)
        #print('effectorCurrent-', effectorCurrent)

        apf.reset()  #  reset
        rewardSum = 0
        stateBefore = [None, None, None, None, None, None, None, None, None]  # first 6 : joints; last 3 : effector coordinate
        thetaBefore = [None, None, None, None, None, None]
        effectorBefore = [None, None, None]
        #print('stateBefore-', stateBefore)
        #print('thetaBefore-', thetaBefore)
        #print('effectorBefore-', effectorBefore)

        # ------------- joint path
        path1 = apf.startTheta.copy()
        path2 = apf.startEffector.copy()

        collision = 0
        step_count = 0

        loss_pi_sum = 0
        loss_q_sum = 0


        for j in range(MAX_STEP):
            step_count += 1
            obsDicq, _ = apf.calculateDynamicState(effectorCurrent)
            obs_sphere = obsDicq['sphere'] #, obsDicq['cylinder'], obsDicq['cone'] #, obs_cylinder, obs_cone
            obs_mix = obs_sphere #+ obs_cylinder + obs_cone
            obs = np.array([])
            for k in range(len(obs_mix)):
                obs = np.hstack((obs, obs_mix[k]))
            #print('obs:' , obs)
            if episode > 50:

                action = agent.get_action(obs, noise_scale=noise)
                #print('action  epi>100:', action)
                #action_sphere = action[0:apf.numberOfSphere]
                #action_cylinder = action[apf.numberOfSphere:apf.numberOfSphere + apf.numberOfCylinder]

            else:
                action = [random.uniform(act_bound[0], act_bound[1]) for k in range(6)]
                #action_cylinder = [random.uniform(act_bound[0],act_bound[1]) for k in range(apf.numberOfCylinder)]
                #action = action_sphere + action_cylinder + action_cone


            # interact with environment---------------------------------------------------------------------------------------------

            stateNext, thetaNext, effectorNext = apf.getNextState(action, thetaCurrent)

            path1 = np.row_stack((path1, thetaNext))
            path2 = np.row_stack((path2, effectorNext))

            obsDicqNext, nearestObs = apf.calculateDynamicState(effectorNext)
            obs_sphere_next = obsDicqNext['sphere']
            # obs_mix_next = obs_sphere_next + obs_cylinder_next + obs_cone_next
            obs_next = np.array([])
            for k in range(len(obs_sphere_next)):
                obs_next = np.hstack((obs_next, obs_sphere_next[k]))
            #print('obs_next:', obs_next)

            flag = apf.sphere_collision_check(Joint_point(thetaNext))
            #print('flag:', flag)
            if flag[0] == 0:
                collision += 1
            optDirection = apf.getUnitCompositeForce(effector=effectorCurrent, eta1List=apf.eta0, epsilon=apf.epsilon0, nearestObstacle=nearestObs)
            #print('optDirection:',optDirection)

            reward = apf.getRewardA(apf, flag, effectorNext, effectorCurrent, optDirection)

            rewardSum += reward

            done = True if apf.distanceCost(apf.goal, effectorNext) < apf.threshold else False
            agent.replay_buffer.store(obs, action, reward, obs_next, done)

            if episode >= 50 and j % update_every == 0:
                if agent.replay_buffer.size >= batch_size:
                    update_cnt += update_every
                    for _ in range(update_every):
                        batch = agent.replay_buffer.sample_batch(batch_size)

                        loss_q, loss_pi = agent.update(data=batch)

                        loss_pi_sum += loss_pi
                        loss_q_sum += loss_q


                        # record the loss, loss_q, loss_pi
                        #loss_pi_List.append(loss_pi)
                        #if episode > 51:
                            #loss_q_List.append(loss_q)


            if done:
                if collision == 0:
                    feasible_path_steps.append(step_count)
                    feasible_path_episode.append(episode)
                    with open("figure/feasible_path.txt", "a") as feasible_path:
                        feasible_path.write(''.join(str(path1)) + '\n')

                break

            # update the states, including Theta and effector
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
            #print('effectorCurrent--', effectorCurrent)



        print('Episode:', episode, 'collision:', collision, 'Reward:%f' % rewardSum, 'noise:%f' % noise, 'steps:%f' %step_count)
        rewardList.append(round(rewardSum,2))

        loss_pi_List.append(loss_pi_sum*0.005)
        loss_q_List.append(loss_q_sum*0.005)

        if rewardSum > maxReward and collision == 0:
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Reward is the best so far! Model is saved!')
            maxReward = rewardSum
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& rewardSum', rewardSum)
            print('new best path1:', path1)
            print('new best path2:', path2)
            print('new best goaling:', effectorNext)
            torch.save(agent.ac.pi, 'trained_models/434.pkl')

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
    #plt.show()
    plt.savefig(fig_path)
    plt.close()


    idx_loss_q = [i for i in range(1, len(loss_q_List) + 1)]
    fig_path = 'figure/' + 'loss_q' + '.png'
    plt.plot(idx_loss_q, loss_q_List)
    plt.xlabel('episode')
    plt.ylabel('loss Q')
    plt.title('loss of Q network')
    #plt.show()
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
    plotfile.write('\n')
    plotfile.writelines(str(loss_pi_List))
    plotfile.write('\n')
    plotfile.writelines(str(loss_q_List))
