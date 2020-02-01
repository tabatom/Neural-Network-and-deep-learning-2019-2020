import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Open test file
data_test = pd.read_csv("test_set.txt", header=None).values
x_test = data_test[:,0]
y_test = data_test[:,1]


#%% ---------- Activation functions ----------

# Define activation function: SIGMOID
from scipy.special import expit
sigmoid = expit
# 1st derivative
d_sigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))


# Define activation function: RELU
def ReLU(x):
    x1 = np.array(x)
    x1[x1<0] = 0
    return x1
# 1st derivative
def d_ReLU(x):
    x1 = np.array(x)
    x1[x1>=0] = 1
    x1[x1<0] = 0
    return x1


# Define activation function: ELU
def ELU(x, alpha=1):
    x1 = np.array(x)
    x1[x1<0] = alpha*(np.exp(x1[x1<0]) - 1)
    return x1
# 1st derivative
def d_ELU(x, alpha=1):
    x1 = np.array(x)
    x1[x1>=0] = 1
    x1[x1<0] = alpha*np.exp(x1[x1<0])
    return x1


#%% ---------- Network class ----------

class Network():
    
    def __init__(self, Ni, Nh1, Nh2, No, act_func, d_act_func):
            
        ### WEIGHT INITIALIZATION (Xavier)
        # Initialize hidden weights and biases (layer 1)
        Wh1 = (np.random.rand(Nh1, Ni) - 0.5) * np.sqrt(12 / (Nh1 + Ni))
        Bh1 = np.zeros([Nh1, 1])
        self.WBh1 = np.concatenate([Wh1, Bh1], 1) # Weight matrix including biases
        # Initialize hidden weights and biases (layer 2)
        Wh2 = (np.random.rand(Nh2, Nh1) - 0.5) * np.sqrt(12 / (Nh2 + Nh1))
        Bh2 = np.zeros([Nh2, 1])
        self.WBh2 = np.concatenate([Wh2, Bh2], 1) # Weight matrix including biases
        # Initialize output weights and biases
        Wo = (np.random.rand(No, Nh2) - 0.5) * np.sqrt(12 / (No + Nh2))
        Bo = np.zeros([No, 1])
        self.WBo = np.concatenate([Wo, Bo], 1) # Weight matrix including biases
        
        ### ACTIVATION FUNCTION
        self.act = act_func
        self.act_der = d_act_func
        
    # Function to make prediction
    def forward(self, x, additional_out=False):
        
        # Convert to numpy array
        x = np.array(x)
        
        ### Hidden layer 1
        # Add bias term
        X = np.append(x, 1)
        # Forward pass (linear)
        H1 = np.matmul(self.WBh1, X)
        # Activation function
        Z1 = self.act(H1)
        
        ### Hidden layer 2
        # Add bias term
        Z1 = np.append(Z1, 1)
        # Forward pass (linear)
        H2 = np.matmul(self.WBh2, Z1)
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        Z2 = np.append(Z2, 1)
        # Forward pass (linear)
        Y = np.matmul(self.WBo, Z2)
        # NO activation function
        
        if additional_out:
            return Y.squeeze(), Z2
        
        return Y.squeeze()
        
    def update(self, x, label, lr):
        
        # Convert to numpy array
        X = np.array(x)
        
        ### Hidden layer 1
        # Add bias term
        X = np.append(X, 1)
        # Forward pass (linear)
        H1 = np.matmul(self.WBh1, X)
        # Activation function
        Z1 = self.act(H1)
        
        ### Hidden layer 2
        # Add bias term
        Z1 = np.append(Z1, 1)
        # Forward pass (linear)
        H2 = np.matmul(self.WBh2, Z1)
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        Z2 = np.append(Z2, 1)
        # Forward pass (linear)
        Y = np.matmul(self.WBo, Z2)
        # NO activation function
        
        # Evaluate the derivative terms
        D1 = Y - label
        # Adding L2 regularization (alpha=1/2, so when differentiating there is a "1" factor)
        # Need the norm of the wheghts
        #D1 = Y - label + np.norm(WBo[:,:-1])
        D2 = Z2
        D3 = self.WBo[:,:-1]
        D4 = self.act_der(H2)
        D5 = Z1
        D6 = self.WBh2[:,:-1]
        D7 = self.act_der(H1)
        D8 = X
        
        # Layer Error
        Eo = D1
        Eh2 = np.matmul(Eo, D3) * D4
        Eh1 = np.matmul(Eh2, D6) * D7
        

        # Derivative for weight matrices
        dWBo = np.matmul(Eo.reshape(-1,1), D2.reshape(1,-1))
        dWBh2 = np.matmul(Eh2.reshape(-1,1), D5.reshape(1,-1))
        dWBh1 = np.matmul(Eh1.reshape(-1,1), D8.reshape(1,-1))
        
        # Update the weights
        self.WBh1 -= lr * dWBh1
        self.WBh2 -= lr * dWBh2
        self.WBo -= lr * dWBo
        
        # Evaluate loss function
        loss = (Y - label)**2/2
        
        return loss
    
    def plot_weights(self):
    
        fig, axs = plt.subplots(3, 1, figsize=(12,6))
        axs[0].hist(self.WBh1.flatten(), 20)
        axs[1].hist(self.WBh2.flatten(), 50)
        axs[2].hist(self.WBo.flatten(), 20)
        plt.legend()
        plt.grid()
        plt.show()



# The best parameters in my case were:
#	Nh1		= 200
#	Nh2		= 200
#	NN_act_f	= ReLU
#	Nh_d_act_f	= d_ReLU
# While Ni = 1 and No = 1 are standard...

# ---------- Initializing the network ----------
net = Network(Ni=1, Nh1=200, Nh2=200, No=1, act_func=ELU, d_act_func=d_ELU)

# Loading wheights
net.WBo = np.load("WBo.npy")
net.WBh1 = np.load("WBh1.npy")
net.WBh2 = np.load("WBh2.npy")

y_test_est = np.array([net.forward(x) for x in x_test])

# np.savetxt("test_set.txt", y_test_est)

MSE = np.mean((y_test - y_test_est)**2/2)

print(MSE)
