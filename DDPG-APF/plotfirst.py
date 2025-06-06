import numpy as np
import matplotlib.pyplot as plt

instances = ['(1)','(2)','(3)','(4)','(5)','(6)','(7)','(8)','(9)','(10)','(11)','(12)']


# average time

x = np.arange(len(instances))
width = 0.2
d = x
p = x + width
t = x + 2*width
apf = x - width

model = ['DDPG_APF', 'pure_DDPG', 'TD3_APF']
time = [1.083, 1.276, 1.489]
plt.bar(model, time, width=width, color='blue', label='DDPG_APF')
for i in range(len(model)):
    plt.text(model[i], time[i], time[i], va="bottom",ha="center",fontsize=8)
title1 = 'Average time for one episode'
plt.title(title1)
plt.xlabel('Agents')
plt.ylabel('Mean time over 700 episodes/ second')
fig_path = 'result/' + 'average time' + '.png'
plt.savefig(fig_path)
plt.close()


DDPG = [86,119,180,236,113,149,248,125,150,168,156,415]  #[86,162,125,150,168,156,236,248,113,180,415,149]
TD3 = [189,162,194,257,576,629,660,700,700,700,700,700]  #[189,119,700,700,700,700,227,660,160,174,700,629]
pure = [133,418,259,284,700,182,222,170,227,160,196,585] #[133,418,170,227,160,196,284,222,700,259,585,182]
plt.bar(d, DDPG, width=width, color='crimson', label='DDPG_APF')
plt.bar(p, pure, width=width, color='darkgray', label='pure DDPG')
plt.bar(t, TD3, width=width, color='lightseagreen', label='TD3_APF')
plt.xticks(x+width, labels=instances)
for i in range(len(instances)):
    plt.text(d[i], DDPG[i], DDPG[i], va="bottom", ha="center", fontsize=6)
    plt.text(t[i], TD3[i], TD3[i], va="bottom", ha="center", fontsize=6)
    plt.text(p[i], pure[i], pure[i], va="bottom", ha="center", fontsize=6)
plt.legend(loc='upper left', prop={'size':7})
title2 = 'Episode of the first feasible solution'
plt.title(title2)
plt.xlabel('Representative task instances')
plt.ylabel('First feasible solution:y-th episode')
fig_path = 'result/' + 'first feasible' + '.png'
plt.savefig(fig_path)
plt.close()
















