import sys
import getopt
import numpy as np
#from keras.datasets import fashion_mnist 
from matplotlib import pyplot as plt

#TAKING INPUTS ARGUMENTS AND SETTING DEFAULT VALUES
wandb_project =""
wandb_entity = ""
dataset = "fashion_mnist"
epochs = 1
batch_size = 4
loss = "cross_entropy"
opimizer = "sgd"
learning_rate = 0.1
momentum = 0.5
beta = 0.5
beta1 = 0.5
beta2 = 0.5
epsilon = 1e-6
weight_decay = 0.0
weight_init = "random"
num_layers = 1
hidden_size = 4
activation = "sigmoid"
argv = sys.argv[1:]
try:
    options, args = getopt.getopt(argv, "wp:we:d:e:b:l:o:lr:m:beta:beta1:beta2:eps:w_d:w_i:nhl:sz:a",     #shortpopts for greater than one character isnt working
                               ["wandb_project=",
                                "wandb_entity=",
                                "dataset=",
                                "epochs=",
                                "batch_size=",
                                "loss=",
                                "optimizer=",
                                "learning_rate=",
                                "momentum=",
                                "beta=",
                                "beta1=",
                                "beta2=",
                                "epsilon=",
                                "weight_decay=",
                                "weight_init=",
                                "num_layers=",
                                "hidden_size=",
                                "activation="])
except:
    print("Error message") #when does this get executed?

for name, value in options:
    #print(name)
    """print("last character", name[-1])
    print(type(name), len(name), len("--epochs"))
    print(name > "--epochs")"""
    if name in ['-wp', '--wandb_project']:
        wandb_project = value
    elif name in ['-we', '--wandb_entity']:
        wandb_entity = value
    elif name in ['-d', '--dataset']:
        dataset = value
    elif name in ['-e', '--epochs']:
        epochs = int(value)
    elif name in ['-b', '--batch_size']:
        batch_size = int(value)
    elif name in ['-l', '--loss']:
        loss = value
    elif name in ['-o', '--optimizer']:
        optimizer = value
    elif name in ['-lr', '--learning_rate']:
        learning_rate = float(value)
    elif name in ['-m', '--momentum']:
        momentum = float(value)
    elif name in ['-beta', '--beta']:
        beta = float(value)
    elif name in ['-beta1', '--beta1']:
        beta1 = float(value)
    elif name in ['-beta2', '--beta2']:
        beta2 = float(value)
    elif name in ['eps', 'epsilon']:
        epsilon = float(value)
    elif name in ['-w_d', '--weight_decay']:
        weight_decay = float(value)
    elif name in ['-w_i', '--weight_init']:
        weight_init = value
    elif name in ['-nhl', '--num_layers']:
        num_layers = int(value)
    elif name in ['-sz', '--hidden_size']:
        hidden_size = int(value)
    elif name in ['-a', '--activation']:
        activation = value
    else: 
        print("INVALID ARGUMENT NAME")

print ("Number of arguments:", len(sys.argv), "arguments")
print ("Argument List:", str(sys.argv))
""" checking input arguments
print("dataset = ", dataset)
print("loss = ", loss)
print("learning_rate = ", learning_rate)
print("epochs = ", epochs + 4)
print(wandb_project)"""
#WRITE CODE TO CHECK IF THE ARGUMENTS ARE TAKING LEGAL VALUES (ACTIVATION FUNCTIONS IN THE ONES WE HAVE DEFINED ONLY OR NOT)

#------------------------------------------------------------------------------------------------------------------------------------------

#QUESTION 1
#LOADING AND DISPLAYING THE PICTURES IN THE SELECTED DATASET 
print(dataset)
exec(f"from keras.datasets import {dataset} as ds")
(train_X, train_y), (test_X, test_y) = ds.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

plt.figure()
for i in range(9):  
    plt.subplot(330 + 1 + i)
    plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------

#QUESTION 2
#IMPLEMENTING A FEEDFORWARD NEURAL NETWORK WITH 784 inputs (pixels) AND 10 outputs (classification probabilities)
#NUMBER OF HIDDEN LAYERS IS num_layers 
#NUMBER OF NEURONS IN EACH HIDDEN LAYER IS hidden_size (same number of neurons in each hidden layer)

#Creating a weights lists which has weights for all the layers
Weights = [np.random.normal(0,1,(hidden_size,784))] #number of inputs 
intermediate_weights = [np.random.normal(0,1,(hidden_size, hidden_size))]*(num_layers-1)
Weights = Weights + intermediate_weights
Weights = Weights + [np.random.normal(0,1,(10, hidden_size))] #number of output classes

#Similarly for biases
Biases = [np.random.normal(0,1,(hidden_size))]*num_layers
Biases = Biases + [np.random.normal(0,1,(10))] #number of output classes
#print(num_layers, hidden_size)
#print("sssss",len(Biases))

def activationFunc(x, activation = 'sigmoid'):
    if activation == 'identity':
        return x
    elif activation == 'sigmoid':
        return 1/(1 + np.exp(-x))   
    elif activation == 'tanh':
        return np.tanh(x)
    elif activation == 'ReLU':
        return np.maximum(0, x)
    else: 
        return "INVALID ACTIVATION FUNCTION"
    

def forward_prop(inputs, Weights, Biases, num_layers, hidden_size, activation): 
    a = []
    h = []
    inputs = inputs/255
    #bleh = bleh/255
    #print(inputs == bleh)
    #print(inputs)
    a.append(np.matmul(Weights[0], inputs) + Biases[0])
    h.append(activationFunc(a[0],activation))
    #print(a,h)
    for i in range(1,num_layers):
        a.append(np.matmul(Weights[i], h[-1]) + Biases[i])
        h.append(activationFunc(a[-1],activation))
    a.append(np.matmul(Weights[-1],h[-1]) + Biases[-1])
    y_pred = np.exp(a[-1])/(np.sum(np.exp(a[-1])))
    #print("zzzzzzzzzzzzzzzz",y_pred)
    #print(bluh == y_pred)
    return a,h,y_pred

"""a,h,y = forward_prop(train_X[0].flatten(), Weights, Biases, num_layers, hidden_size)
print("Aaa",a,"Hhh",h,"Yyy",y) checking"""

def back_prop(a, h, inputs, Weights, y_pred , y_true, num_layers, loss, activation):  #y_pred is probab vector, #y_true is value #actually i dont think i need a vector here
    #Finding grad of loss wrt output layer depending upon which loss
    inputs = inputs/255
    if loss == "mean_squared_error":
        grad_aL = np.matmul(np.array([np.eye(10)[i]*y_pred - y_pred*y_pred[i] for i in range(10)]),2*(y_pred - y_true)) #VERIFY
    elif loss == "cross_entropy": 
        grad_aL = -(np.eye(10)[y_true] - y_pred)
    #print(np.shape(grad_aL))
    #Finding g'(z) depending upon activation function    
    h = np.array(h)
    gdot_a = np.ones(np.size(h)) #default identity
    if activation == "sigmoid":
        gdot_a = (1-h)*h
    elif activation == "tanh":
        gdot_a = 1 - np.power(h,2)
    elif activation == "ReLU":
        gdot_a[np.where(h > 0)] = 1
    
    #storing only gradients wrt weights and biases
    gradW = [0]*(num_layers + 1)
    gradB = [0]*(num_layers + 1)
    grad_ak = grad_aL 
    #print(grad_ak)
    #print(num_layers)
    for i in range(num_layers, 0 , -1): 
        #print("back_prop",i)
        gradW[i] = np.outer(grad_ak, h[i-1])
        gradB[i] = grad_ak
        #print(np.shape(Weights[i]), np.shape(grad_ak))
        grad_ak = np.matmul(Weights[i].T, grad_ak) * gdot_a[i-1]
    gradW[0] = np.outer(grad_ak, inputs)
    gradB[0] = grad_ak
    return gradW, gradB
    

def grad_descent(train_X, train_y, epochs, learning_rate, Weights, Biases, num_layers, hidden_size, loss, activation): #need weights, biases, learning_rate, epochs
    for j in range(epochs):
        print("epochs",j)
        a, h, y_pred = forward_prop(train_X[0].flatten(), Weights, Biases, num_layers, hidden_size, activation)
        grad_W, grad_B = back_prop(a, h,train_X[0].flatten(), Weights, y_pred, train_y[0], num_layers, loss, activation)
        for i in range(1,np.shape(train_X)[0]):
            a, h, y_pred = forward_prop(train_X[i].flatten(), Weights, Biases, num_layers, hidden_size, activation)
            gradW, gradB = back_prop(a, h, train_X[0].flatten(), Weights, y_pred, train_y[i], num_layers, loss, activation)
            grad_W[0] += gradW[0]
            grad_B[0] += gradB[0]
            grad_W[-1] += gradW[-1]
            grad_B[-1] += gradB[-1]
            grad_W[1:-1] = np.array(grad_W[1:-1]) + np.array(gradW[1:-1])
            grad_B[1:-1] = np.array(grad_B[1:-1]) + np.array(gradB[1:-1])
        print("grad_W",grad_W, "grad_B", grad_B)
        Weights[0] = Weights[0] - learning_rate*grad_W[0]
        Biases[0] = Biases[0] - learning_rate*grad_B[0]
        Weights[-1] = Weights[-1] - learning_rate*grad_W[-1]
        Biases[-1] = Biases[-1] - learning_rate*grad_B[-1]
        Weights[1:-1] = np.array(Weights[1:-1]) - learning_rate*np.array(grad_W[1:-1])
        Biases[1:-1] = np.array(Biases[1:-1]) - learning_rate*np.array(grad_B[1:-1])
    return Weights, Biases

def sgd(train_X, train_y, epochs, learning_rate, Weights, Biases, num_layers, hidden_size, loss, activation): #need weights, biases, learning_rate, epochs
    for j in range(epochs):
        print("epochs",j)
        a, h, y_pred = forward_prop(train_X[0].flatten(), Weights, Biases, num_layers, hidden_size, activation)
        grad_W, grad_B = back_prop(a, h,train_X[0].flatten(), Weights, y_pred, train_y[0], num_layers, loss, activation)
        for i in range(1,np.shape(train_X)[0]):
            a, h, y_pred = forward_prop(train_X[i].flatten(), Weights, Biases, num_layers, hidden_size, activation)
            gradW, gradB = back_prop(a, h, train_X[0].flatten(), Weights, y_pred, train_y[i], num_layers, loss, activation)
            grad_W[0] += gradW[0]
            grad_B[0] += gradB[0]
            grad_W[-1] += gradW[-1]
            grad_B[-1] += gradB[-1]
            grad_W[1:-1] = np.array(grad_W[1:-1]) + np.array(gradW[1:-1])
            grad_B[1:-1] = np.array(grad_B[1:-1]) + np.array(gradB[1:-1])
        #print("grad_W",grad_W, "grad_B", grad_B)
            Weights[0] = Weights[0] - learning_rate*grad_W[0]
            Biases[0] = Biases[0] - learning_rate*grad_B[0]
            Weights[-1] = Weights[-1] - learning_rate*grad_W[-1]
            Biases[-1] = Biases[-1] - learning_rate*grad_B[-1]
            Weights[1:-1] = np.array(Weights[1:-1]) - learning_rate*np.array(grad_W[1:-1])
            Biases[1:-1] = np.array(Biases[1:-1]) - learning_rate*np.array(grad_B[1:-1])
    return Weights, Biases
##--------------------------------------------------------------------------------------------------------------------------------------
## BRINGING IT ALL TOGETHER FOR THE FIRST TIME

print("W", Weights, "B", Biases)
#Weights, Biases = grad_descent(train_X, train_y, epochs, learning_rate, Weights, Biases, num_layers, hidden_size, loss, activation)
Weights, Biases = sgd(train_X, train_y, epochs, learning_rate, Weights, Biases, num_layers, hidden_size, loss, activation)
print("W", Weights, "B", Biases)
y_pred = [np.zeros(10)]
for i in range(np.shape(test_X)[0]):
    #print(i)
    y_pred.append(forward_prop(test_X[i].flatten(), Weights, Biases, num_layers, hidden_size, activation)[2])
    #print(np.argmax(y_pred[-1]))
    #print(y_pred[-1] == y_pred[-2])

y_pred = np.array(y_pred[1:])
print(y_pred)
y_pred = np.argmax(y_pred, 1)
print(y_pred)
print(np.size(np.where(y_pred == test_y))/np.size(test_y))
print(test_y)





