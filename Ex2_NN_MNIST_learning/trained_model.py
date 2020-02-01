import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io
import skorch

import torch
import torch.nn as nn
import torch.optim as optim


# Loading data
MNIST_ = scipy.io.loadmat('MNIST.mat')

x_test = MNIST_['input_images'].copy()
y_test = MNIST_['output_labels'].copy()


# ---------- Neural Network model ----------

### Define the network class
class Net(nn.Module):
    
    def __init__(self, Ni, Nh1, Nh2, No, act_func):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, No)
        
        self.act = act_func()
        
        
    def forward(self, x, additional_out=False):
        
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        out = self.fc3(x)
        out = nn.functional.softmax(out, dim=-1)
        
        if additional_out:
            return out, np.argmax(out)
        
        return out



# Retrieving model
MY_best_net = Net(784, 96, 24, 10, nn.modules.activation.Tanh)
MY_best_net.load_state_dict(torch.load("my_best_net.pkl"))

test_label = MY_best_net(torch.tensor(x_test)).float().detach().numpy()
#print(test_label)
test_label = np.argmax(test_label, axis=1)
#print(test_label)
#for ii in range(len(test_label)):
#    print(test_label[ii], y_test[ii])
print("Model accuracy:\t", np.mean(test_label==y_test.squeeze()))
