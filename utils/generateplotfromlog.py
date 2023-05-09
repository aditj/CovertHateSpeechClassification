import matplotlib.pyplot as plt
### Read lines from cmd.log and plot the accuracies
accuracies = []
i = 0
with open("cmd.log") as file:
    for line in file:
        i += 1
        if "Accuracies" in line:
            ## append the float after the string "Accuracies: " and before " "
            accuracies.append(float(line.split("Accuracies: ")[1].split(" ")[0]))
        else:
            if len(accuracies) > 0:
                accuracies.append(accuracies[-1])
            else:
                accuracies.append(0)

plt.plot(range(i),accuracies)
plt.xlabel("Communication Round")
plt.ylabel("Accuracies of the Aggregated Model")
plt.savefig("../plots/accuracies.png")
