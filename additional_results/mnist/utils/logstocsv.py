# Convert logs to csv files
import pandas as pd
filename = "./data/logs/cummulative_experiment_results.txt"
output_filename = "./data/logs/cumm_var_reduce.csv"

rows = [   ]
with open(filename, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace("\n","")
        date,conditions,accuracies = line.split(": ")
        conditions = conditions.split("_")
        conditions.remove("acc")
        if len(conditions) > 7:
            conditions.remove("MNIST")
                
        accuracies = accuracies.split(",")

        if len(accuracies) <3:
            accuracies.append(None)
        row = [date] + conditions + accuracies
        if len(row)>11:
            print(row)
        rows.append(row)
<<<<<<< HEAD
pd.DataFrame(rows,columns=["date","n_dev","n_succ","n_rounds","prop_data_eav","n_class_eav","class_dist_eav","greedy","eaves_acc","smart_eaves_accuracy","learner_acc","learning_queries"]).to_csv(output_filename,index=False)
=======
pd.DataFrame(rows,columns=["date","n_dev","n_succ","n_rounds","prop_data_eav","n_class_eav","class_dist_eav","greedy","eaves_acc","smart_eaves_accuracy","learner_acc"]).to_csv(output_filename,index=False)
#pd.DataFrame(rows,columns=['date',"n_dev","n_succ","n_rounds",""])
>>>>>>> 17e9b34c192e8e1074c1e86fc2221de7ff2c3d85
