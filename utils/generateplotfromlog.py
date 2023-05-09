import matplotlib.pyplot as plt
import seaborn as sns
### Read lines from cmd.log and plot the accuracies
accuracies = []
eavesdropper_accuracies = []
f1_scores = []
eavesdropper_f1_scores = []
balanced_accuracies = []
eavesdropper_balanced_accuracies = []

i = 0
with open("./data/cmd.log") as file:
    for line in file:
        i += 1
        if "Eavesdropper" in line:
            eavesdropper_accuracies.append(float(line.split("accuracy: ")[1].split(" ")[0].split(",")[0]))
            # Extract F1 score from INFO:root:Communication round 3 Eavesdropper accuracy: 0.8007623007623008 F1 0.8007623007623008 Balanced Accuracy 0.8030884949284756
            eavesdropper_f1_scores.append(float(line.split("F1 ")[1].split(" ")[0]))
            # Extract Balanced Accuracy from INFO:root:Communication round 3 Eavesdropper accuracy: 0.8007623007623008 F1 0.8007623007623008 Balanced Accuracy 0.8030884949284756
            eavesdropper_balanced_accuracies.append(float(line.split("Balanced Accuracy ")[1].split(" ")[0]))
        else:
            if len(eavesdropper_accuracies) > 0:
                eavesdropper_accuracies.append(eavesdropper_accuracies[-1])
                eavesdropper_f1_scores.append(eavesdropper_f1_scores[-1])
                eavesdropper_balanced_accuracies.append(eavesdropper_balanced_accuracies[-1])
            else:
                eavesdropper_accuracies.append(0)
                eavesdropper_f1_scores.append(0)
                eavesdropper_balanced_accuracies.append(0)
        if "Accuracies" in line:
                
            ## append the float after the string "Accuracies: " and before " "
            accuracies.append(float(line.split("Accuracies: ")[1].split(" ")[0].split(",")[0]))
            # Extract 2nd float from INFO:root:Communication round: 1, Aggregated Accuracies: 0.5235701906412478, 0.410044438170232, 0.05815736902186073
            f1_scores.append(float(line.split("Accuracies: ")[1].split(" ")[1].split(",")[0]))
            # Extract 3rd float from INFO:root:Communication round: 1, Aggregated Accuracies: 0.5235701906412478, 0.410044438170232, 0.05815736902186073
            balanced_accuracies.append(float(line.split("Accuracies: ")[1].split(" ")[2].split(",")[0]))
        else:
            if len(accuracies) > 0:
                accuracies.append(accuracies[-1])
                f1_scores.append(f1_scores[-1])
                balanced_accuracies.append(balanced_accuracies[-1])
            else:
                accuracies.append(0)
                f1_scores.append(0)
                balanced_accuracies.append(0)

## Latex 
#plt.rc('text', usetex=True)
sns.set_style("darkgrid")
sns.set_context("paper")

plt.plot(range(i),eavesdropper_accuracies,label="Eavesdropper Score")
plt.plot(range(i),accuracies,label="Aggregated Score")
plt.plot(range(i),f1_scores,label="Aggregated Model F1")
plt.plot(range(i),eavesdropper_balanced_accuracies,label="Eavesdropper Balanced Accuracy")
plt.plot(range(i),eavesdropper_f1_scores,label="Eavesdropper F1")
plt.plot(range(i),balanced_accuracies,label="Aggregated Model Balanced Accuracy")
plt.xlabel("Communication Round",fontsize=14)
plt.ylabel("Accuracy Score",fontsize=14)
plt.legend()
plt.savefig("./data/plots/accuracies.pdf")
