import torch
from models import BERTClass
from eavesdropper import Eavesdropper
import numpy as np
def check_parameters_change(parameters_before,parameters_after):
    for layer in parameters_before:
        if torch.equal(parameters_before[layer],parameters_after[layer]):
            continue
        else:
            print("Parameters changed")
            return True
    print("Parameters did not change")
    return False
# test if the parameters of eavesdropper change across training loop 
e = Eavesdropper(32,10,200,1,1e-5,n_classes=1)
for i in range(10):
    parameters_before = e.get_parameters()
    e.train(None)
    e.evaluate()
    parameters_after = e.get_parameters()
    check_parameters_change(parameters_before,parameters_after)
