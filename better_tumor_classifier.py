

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


wb = pd.read_csv('Cancer.csv')

wb = wb.replace('?',0)

wb_np = np.array(wb)
score = wb_np[:,10]
score = score.reshape(699,1)
wb_np = wb_np[:,1:10]


train_set_X = wb_np[0:550]
train_set_Y = score[0:550]
test_set_X = wb_np[550:]
test_set_Y = score[550:]

train_set_Y = np.where(train_set_Y==2,0,1)
test_set_Y = np.where(test_set_Y==2,0,1)

def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))

def initialize_parameters_he(layers_dims):

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
     
    for l in range(1, L):
       
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2.0/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        
        
    return parameters

def initialize_adam(parameters) :

    L = len(parameters) // 2 
    v = {}
    s = {}
    
    
    for l in range(L):

        v["dW" + str(l+1)] = np.zeros((parameters['W'+str(l+1)].shape))
        v["db" + str(l+1)] = np.zeros((parameters['b'+str(l+1)].shape))
        s["dW" + str(l+1)] = np.zeros((parameters['W'+str(l+1)].shape))
        s["db" + str(l+1)] = np.zeros((parameters['b'+str(l+1)].shape))

    return v, s

def forward_activation(x,W,b):
    
    op = np.dot(W,x)+b
    op = np.where(op<0,0,op) # relu
        
    return op

def forward_propagation(train_x,parameters,layers):
    
    a = train_x
    cache = {}
    for i in range(layers-1):
        a = forward_activation(a,parameters['W'+str(i+1)],parameters['b'+str(i+1)])
        cache['da'+str(i+1)] = a
    
    a = np.dot(parameters['W'+str(layers)],a)+parameters['b'+str(layers)]
    a = sigmoid(a)
    cache['A'] = a
    
    return a,cache

def compute_cost_with_regularization(train_y, Y, parameters, lambd):

    m = Y.shape[1]
    L = len(parameters) // 2
    
    L2_regularization_cost = 0
    
   
    cross_entropy_cost = -(np.sum(np.dot(np.log(train_y),Y.T) + np.dot(np.log(1-train_y),1-Y.T)))/m 
    
    for l in range(L):
        L2_regularization_cost = L2_regularization_cost + np.sum(np.square(parameters['W'+str(l+1)]))
    
    L2_regularization_cost = (lambd * L2_regularization_cost )/(2*m)
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

def backward_propagation_with_regularization(X, Y, cache, lambd):
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    
    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd*(W3))/m
    
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    
    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd*(W2))/m
    
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
   
    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd*(W1))/m
    
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

def backward_propagation_with_L2(parameters,cache,train_y,train_x,lambd):
    
    grads = {}
    m = train_y.shape[1]
    L = len(cache)
    dz = cache['A'] - train_y
    
    qw = (np.dot(dz,cache['da'+str(L-1)].T))/m
   
    grads['dw'+str(L)] = qw + (lambd*(parameters['W'+str(L)]))/m
    aq = (np.sum(dz,axis = 1, keepdims = True)/m)
    grads['db'+str(L)] = aq
    
    
    for i in range(L-1,1,-1):
        dz = np.multiply(np.dot(parameters['W'+str(i+1)].T,dz),np.int64(cache['da'+str(i)]>0))
        grads['dw'+str(i)] = (np.dot(dz,cache['da'+str(i-1)].T)/m) + (lambd*(parameters['W'+str(i)]))/m
        grads['db'+str(i)] = (np.sum(dz,axis = 1, keepdims = True)/m)
   
    
    dz = np.multiply(np.dot(parameters['W2'].T,dz),np.int64(cache['da1']>0))
    grads['dw1'] = (np.dot(dz,train_x.T)/m) + (lambd*(parameters['W1']))/m
    grads['db1'] = (np.sum(dz,axis = 1, keepdims = True)/m)
    
    return grads

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
   
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    
    for l in range(L):
        
        v["dW" + str(l+1)] = (beta1*v['dW'+str(l+1)])+((1-beta1)*grads['dw'+str(l+1)])
        v["db" + str(l+1)] = (beta1*v['db'+str(l+1)])+((1-beta1)*grads['db'+str(l+1)])
        
        v_corrected["dW" + str(l+1)] = np.divide(v['dW'+str(l+1)],1-np.power((beta1),t))
        v_corrected["db" + str(l+1)] = np.divide(v['db'+str(l+1)],1-np.power((beta1),t))
        
        s["dW" + str(l+1)] = (beta2*s['dW'+str(l+1)])+((1-beta2)*(np.square(grads['dw'+str(l+1)])))
        s["db" + str(l+1)] = (beta2*s['db'+str(l+1)])+((1-beta2)*(np.square(grads['db'+str(l+1)])))
        
        s_corrected["dW" + str(l+1)] = np.divide(s['dW'+str(l+1)],1-np.power((beta2),t))
        s_corrected["db" + str(l+1)] = np.divide(s['db'+str(l+1)],1-np.power((beta2),t))
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate* (np.divide(v_corrected['dW'+str(l+1)],np.sqrt(s_corrected['dW'+str(l+1)])+epsilon)))
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate* (np.divide(v_corrected['db'+str(l+1)],np.sqrt(s_corrected['db'+str(l+1)])+epsilon)))
        

    return parameters, v, s

def model(X, Y, layers_dims, learning_rate = 0.0007,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True,lambd = 0.00001):
    """
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    np.random.seed(3)                       # For grading purposes, so that your "random" minibatches are the same as ours
    
    # Initialize parameters
    parameters = initialize_parameters_he(layers_dims)
    
    v, s = initialize_adam(parameters)
    
    # Optimization loop
    for i in range(num_epochs):
        

        # Forward propagation
        a, caches = forward_propagation(X, parameters, L-1)

        # Compute cost
        cost = compute_cost_with_regularization(a, Y, parameters, lambd)

        # Backward propagation
        grads = backward_propagation_with_L2(parameters,caches,Y,X,lambd)

        # Update parameters
        t = t + 1 # Adam counter
        parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                       t, learning_rate, beta1, beta2,  epsilon)
        
        # print( the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

parameters = model(train_set_X.T.astype(int), train_set_Y.astype(int).T, [train_set_X.shape[1],8,7,7,7,5,4,3,2,1], learning_rate = 0.003,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5001, print_cost = True)

def classify(test,params):
    a = test
    
    a,b = forward_propagation(a,params,9)
    
    b = np.where(a<0.5,0,1)
    return a,b

a_res,train_res = classify(train_set_X.T.astype(int),parameters)

#for i in range(100):
#       print("index = "+str(i)+"  y = "+str((train_set_Y[i][0]))+"  y^ = "+str((train_res[0][i])))+"  actual = "+str(a_res[0][i])

cnt = 0.0
for i in range(550):
    if train_set_Y[i][0] == train_res[0][i]:
        cnt+=1
cnt = cnt/550
cnt
print ("Accuracy on train",cnt*100)

t_res,test_res = classify(test_set_X.T.astype(int),parameters)

cnt = 0.0
for i in range(149):
    if test_set_Y[i] == test_res[0][i]:
        cnt+=1
cnt = cnt/149
print ("Accuracy on test",cnt*100)

false_negative = 0
false_positive = 0
for i in range(149):
    if test_set_Y[i] == 1 and test_res[0][i] == 0:
        false_negative+=1
    elif test_set_Y[i] == 0 and test_res[0][i] == 1:
        false_positive+=1
print ('False negatives '+str(false_negative)+'\nFalse positives '+str(false_positive))

false_negative = 0
false_positive = 0
for i in range(550):
    if train_set_Y[i][0] == 1 and train_res[0][i] == 0:
        false_negative+=1
    elif train_set_Y[i][0] == 0 and train_res[0][i] == 1:
        false_positive+=1
print ('False negatives '+str(false_negative)+'\nFalse positives '+str(false_positive))

f1_score(train_set_Y[:,0],train_res[0,:],average='binary')

f1_score(test_set_Y,test_res[0,:],average='binary')