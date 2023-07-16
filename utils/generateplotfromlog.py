import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
### Read lines from cmd.log and plot the accuracies

import os 
files = os.listdir("./data/logs/experiment1")
greedy_files = [f for f in files if "True" in f]
optimal_files = [f for f in files if "False" in f]
if len(greedy_files) != len(optimal_files):
    print("Error: number of files not equal")
    exit(0)
N_exp = 11
N_comm = 45
accuracies = np.zeros((N_exp,N_comm))
eavesdropper_accuracies =np.zeros((N_exp,N_comm))
accuracies_greedy =np.zeros((N_exp,N_comm))
eavesdropper_accuracies_greedy = np.zeros((N_exp,N_comm))
i_greedy = 0
i_optimal = 0
with open("./data/logs/experiment1/"+greedy_files[0],"r") as f:
    for j,line in enumerate(f):
        ## Take out communication round from "Communication round: 44, "            
        
        if "--" in line: 
            i_greedy+=1
            continue
        comm_round = int(line.split("Communication round: ")[1].split(",")[0])
        if "Eavesdropper Accuracy" in line:
            eavesdropper_accuracies_greedy[i_greedy,comm_round] = float(line.split(": ")[2].split(",")[0])   
        if "Aggregated Accuracies" in line:
            accuracies_greedy[i_greedy,comm_round] = float(line.split(": ")[2].split(",")[0])

with open("./data/logs/experiment1/"+optimal_files[0],"r") as f:
    for j,line in enumerate(f):
        
        if "--" in line:    
            i_optimal+=1
            continue
        comm_round = int(line.split("Communication round: ")[1].split(",")[0])
        if "Eavesdropper Accuracy" in line:
            eavesdropper_accuracies[i_optimal,comm_round] = float(line.split(": ")[2].split(",")[0])
        if "Aggregated Accuracies" in line:
            accuracies[i_optimal,comm_round] = float(line.split(": ")[2].split(",")[0])
print(i_greedy,i_optimal,N_exp)
assert (i_greedy == i_optimal) and (i_greedy == N_exp)


# replace 0 entries of accuracies,eavesdropper_accuracies,accuracies_greedy,eavesdropper_accuracies_greedy with previous values by looping over
for i in range(N_exp):
    for j in range(N_comm):
        if accuracies[i,j] == 0:
            accuracies[i,j] = accuracies[i,j-1]
        if eavesdropper_accuracies[i,j] == 0:
            eavesdropper_accuracies[i,j] = eavesdropper_accuracies[i,j-1]
        if accuracies_greedy[i,j] == 0:
            accuracies_greedy[i,j] = accuracies_greedy[i,j-1]
        if eavesdropper_accuracies_greedy[i,j] == 0:
            eavesdropper_accuracies_greedy[i,j] = eavesdropper_accuracies_greedy[i,j-1]
print(accuracies)
# print("Accuracies: ",eavesdropper_accuracies_greedy)
sns.set_style("darkgrid")
sns.set_context("paper")
# Plot average and std of accuracies
# take min of accuracies and eavesdropper_accuracies
accuracies_min = np.min(accuracies,axis=0)
accuracies_greedy_min = np.min(accuracies_greedy,axis=0)
eavesdropper_accuracies_min = np.min(eavesdropper_accuracies,axis=0)
eavesdropper_accuracies_greedy_min = np.min(eavesdropper_accuracies_greedy,axis=0)
# take max of accuracies and eavesdropper_accuracies
accuracies_max = np.max(accuracies,axis=0)
accuracies_greedy_max = np.max(accuracies_greedy,axis=0)
eavesdropper_accuracies_max = np.max(eavesdropper_accuracies,axis=0)
eavesdropper_accuracies_greedy_max = np.max(eavesdropper_accuracies_greedy,axis=0)

accuracies = np.mean(accuracies,axis=0)
eavesdropper_accuracies = np.mean(eavesdropper_accuracies,axis=0)
eavesdropper_accuracies_greedy = np.mean(eavesdropper_accuracies_greedy,axis=0)
accuracies_greedy = np.mean(accuracies_greedy,axis=0)
plt.figure(figsize=(8,6))
plt.fill_between(range(N_comm),accuracies_min,accuracies_max,alpha=0.2)
plt.fill_between(range(N_comm),accuracies_greedy_min,accuracies_greedy_max,alpha=0.2)
plt.fill_between(range(N_comm),eavesdropper_accuracies_min,eavesdropper_accuracies_max,alpha=0.2)
plt.fill_between(range(N_comm),eavesdropper_accuracies_greedy_min,eavesdropper_accuracies_greedy_max,alpha=0.2)

# Make the eavesdropper lines dotted
plt.plot(range(N_comm),eavesdropper_accuracies,label="$\mathcal{E}$ acc. under Optimal Policy",linestyle='dashed')
plt.plot(range(N_comm),accuracies,label="$\mathcal{L}$ acc. under Optimal Policy")
plt.plot(range(N_comm),eavesdropper_accuracies_greedy,label="$\mathcal{E}$ acc. under Greedy Policy",linestyle='dashed')
plt.plot(range(N_comm),accuracies_greedy,label="$\mathcal{L}$ acc. under Greedy Policy")

plt.xlabel("Communication Round",fontsize=18)
plt.ylabel("Accuracy Score",fontsize=18)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=16)
plt.savefig("./data/plots/accuracies_10.pdf",bbox_inches='tight',pad_inches=0)
