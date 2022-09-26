import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import numpy as np
from scipy import ndimage

fig = plt.figure()
x = np.arange(0, 102, 2)
prioritized = True
# env_name = "InvertedPendulumBulletEnv-v0"
# env_name = "HopperBulletEnv-v0"
env_name = "Walker2DBulletEnv-v0"
repository = "data/Experiment/%s"
buffer_size = int(1e6)
models = ['','_Transfer']
labels = ['Regular ' + env_name[:-12], 'Transfer ' + env_name[:-12]]
# graph_title = env_name[:-12]+" Buffer Transfer (1e5)"
# graph_title = "Hopper to Walker2d PER Transfer (1e4)"
# graph_title = "Walker2d to Hopper PER Transfer (1e6)"
graph_title = "Hopper to Walker2d Parameters Transfer"
# graph_title = "Walker2d to Hopper Parameters Transfer"
#
if prioritized:
    file_name = "testing_Priority_DDPG_"+env_name[:-12]+"_"+str(buffer_size)
    # file_name = "Priority_DDPG_" + env_name[:-12] + "_" + str(buffer_size)
else:
    file_name = "DDPG_"+env_name[:-12]+"_"+str(buffer_size)

# ---------------------------------- drawing graph ---------------------------------------------------------
area = np.zeros(2)
for i in range(len(labels)):
    data = np.zeros([5,51])
    for j in range(5):
        filename = file_name + models[i] + "_" + str(j) + '.npy'
        print(filename)
        trial_data = np.array(np.load(repository % filename))
        data[j,:] = trial_data
        # print(trial_data)

    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    mean = ndimage.uniform_filter(mean, size=3)
    std = ndimage.uniform_filter(std, size=3)
    plt.plot(x, mean, label=labels[i])
    plt.fill_between(x, mean-std, mean+std, alpha=0.15)
    area[i] = np.trapz(mean, dx=1)

print("Area 1: " + str(area[0]))
print("Area 2: "+str(area[1]))
area_ratio = (area[1] - area[0])/area[0]
print("Area Ratio: "+str(area_ratio))
plt.legend(loc='lower right')
plt.title(graph_title)
plt.xlabel('time steps (1e3)',fontsize=14)
plt.ylabel('average return',fontsize=14)
plt.show()

# --------------------------------- mean and standard deviation -----------------------------------------------
for i in range(len(labels)):
    data = np.zeros([5, 51])
    for j in range(5):
        filename = file_name + models[i] + "_" + str(j) + '.npy'
        trial_data = np.array(np.load(repository % filename))
        data[j,:] = trial_data

    max_data = np.max(data,axis=1)
    print(env_name[:-12]+" "+labels[i])
    print('mean: ',np.mean(max_data))
    print('standard deviation: ', np.std(max_data))

