import os
file_false = "data/logs/experiment1/cache/server_2023-05-15_12:10:16_False.log"
file_true = "data/logs/experiment1/cache/server_2023-05-15_12:10:16_True.log"
with open(file_true) as file_:
    for i in range(5):
        filename = "data/logs/experiment1/exp"+ str(i+10) + "True.log"
        with open(filename, 'w') as f:
            for line in file_:
                if "--" in line:
                    break
                f.write(line)
with open(file_false) as file_:
    for i in range(5):
        filename = "data/logs/experiment1/exp" + str(i+10) + "False.log"
        with open(filename, 'w') as f:
            for line in file_:
                if "--" in line:
                    break
                f.write(line)