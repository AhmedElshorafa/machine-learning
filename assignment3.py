import pandas as pd 
from matplotlib import pyplot
import numpy as np
data = pd.read_csv("house_data_complete.csv")
def gradientDescentMulti(X, y, theta, alpha, num_iters,lamda):
    
    m = y.shape[0]
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for  i  in  range ( num_iters ):
        
        a=alpha/m
        hypothesis=np.dot(X,theta)
        theta=theta*(1-(alpha*lamda)/m)-((a)*(np.dot(X.T,subtract(hypothesis,y))))
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta,lamda))
    
    return theta, J_history

def subtract(a,b):
    array =np.zeros(len(a))
    for i in range(len(a)):
        array[i] = a[i]-b[i]
    return array

def computeCostMulti(X, y, theta,lamda):
    
    m = y.shape[0]  
    J = 0
    i=2 * m
    J= np.dot(subtract(np.dot(X, theta),y), (subtract(np.dot(X, theta),y))) / (i) +((lamda/i)) *np.sum(np.dot(theta,theta))    
    return J


def computeCostcrossV1(X, y, theta):
    
    m = y.shape[0]  
    J = 0
    i=2 * m
    J= np.dot(subtract(np.dot(X, theta),y), (subtract(np.dot(X, theta),y))) / (i)     
    return J
        
def gradientDescentMulti2(X, y, theta, alpha, num_iters,lamda):
    
    m = y.shape[0]
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()   
    J_history  = []
    
    for  i  in  range ( num_iters ):
        
        a=alpha/m
        hypothesis=np.dot(X*X ,theta)
        theta=theta*(1-(alpha*lamda)/m)-((a)*(np.dot(X.T,subtract(hypothesis,y))))
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti2(X, y, theta,lamda))
    
    return theta, J_history


def computeCostMulti2(X, y, theta,lamda):
    
    m = y.shape[0] 
    J = 0
    i=2 * m
    J= np.dot(subtract(np.dot(X*X, theta),y), (subtract(np.dot(X*X, theta),y))) / (i) +((lamda/i)) *np.sum(np.dot(theta,theta))    
    return J

def computeCostcrossV2(X, y, theta):
    
    m = y.shape[0] 
    J = 0
    i=2 * m
    J= np.dot(subtract(np.dot(X*X, theta),y), (subtract(np.dot(X*X, theta),y))) / (i)     
    return J
    
def gradientDescentMulti3(X, y, theta, alpha, num_iters,lamda):
    
   
    m = y.shape[0]
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    J_history  = []
    
    for  i  in  range ( num_iters ):
        
        a=alpha/m
        hypothesis=np.dot(X*X*X ,theta)
        theta=theta*(1-(alpha*lamda)/m)-((a)*(np.dot(X.T,subtract(hypothesis,y))))
        J_history.append(computeCostMulti3(X, y, theta,lamda))
    
    return theta, J_history


def computeCostMulti3(X, y, theta,lamda):
    
    m = y.shape[0] 
    J = 0
    i=2 * m
    J= np.dot(subtract(np.dot(X*X*X, theta),y), (subtract(np.dot(X*X*X, theta),y))) / (i) +((lamda/i)) *np.sum(np.dot(theta,theta))    
    return J
 
def computeCostcrossV3(X, y, theta):
    
    m = y.shape[0] 
    J = 0
    i=2 * m
    J= np.dot(subtract(np.dot(X*X*X, theta),y), (subtract(np.dot(X*X*X, theta),y))) / (i)     
    return J


for i in range (0,3):
    prices=data[["price"]]
    new_prices=prices.to_numpy()
    #print(new_prices.shape)
    data_array =data.to_numpy()
    print(data_array)
    floors=data[["floors"]]
    new_floors=floors.to_numpy()
    print(new_floors)
    fig = pyplot.figure()  # open a new figure
        
    pyplot.plot(new_floors, new_prices, 'ro', ms=10, mec='k')
    pyplot.ylabel('prices')
    pyplot.xlabel('Floors')
    #pyplot.show()
    train_array,cross_validation,testing=np.split(data.sample(frac=1),[int(0.6*len(data)),int(0.8*len(data))]) 
    train_array=data_array[:,4:21] 
    cross_validation=data_array[:, 4:21]
    testing=data_array[:,4:21]

    train_mean =train_array-train_array.mean(axis=0) 
    train_std=np.std(train_array)
    train_array_norm=train_mean/train_std
    new_train_array_norm= np.concatenate([np.ones((train_array_norm.shape[0], 1)), train_array_norm], axis=1)

    cv_mean =cross_validation-cross_validation.mean(axis=0) 
    cross_validation_std=np.std(cross_validation)
    cross_validation_norm=cv_mean/cross_validation_std
    new_cross_validation_norm= np.concatenate([np.ones((cross_validation_norm.shape[0], 1)), cross_validation_norm], axis=1)

    test_mean =testing-testing.mean(axis=0) 
    testing_std=np.std(testing)
    testing_norm=test_mean/testing_std
    new_testing_norm= np.concatenate([np.ones((testing_norm.shape[0], 1)), testing_norm], axis=1)

    lamda=[0.04,0.32,1.32]
    alpha = 0.1
    alpha2 = 0.01
    alpha3= 0.00000000001
    num_iters  = 100
    theta = np.zeros(18)
    theta1_1, J_history_1 = gradientDescentMulti(new_train_array_norm, new_prices, theta, alpha, num_iters,lamda[0])
    theta1_2, J_history_2 = gradientDescentMulti(new_train_array_norm, new_prices, theta, alpha, num_iters,lamda[1])
    theta1_3, J_history_3 = gradientDescentMulti(new_train_array_norm, new_prices, theta, alpha, num_iters,lamda[2])
    hypo1_thetas=np.array([theta1_1,theta1_2,theta1_3])
    cv_1=computeCostcrossV1(new_cross_validation_norm,new_prices,theta1_1)
    cv_2=computeCostcrossV1(new_cross_validation_norm,new_prices,theta1_2)
    cv_3=computeCostcrossV1(new_cross_validation_norm,new_prices,theta1_3)

    cv_array=np.array([cv_1,cv_2,cv_3])
    best_lamda_1=lamda[np.argmin(cv_array,axis=0)]
    best_theta_1=hypo1_thetas[np.argmin(cv_array,axis=0)]
    theta1_4, J_history_4 = gradientDescentMulti2(new_train_array_norm, new_prices, theta, alpha2, num_iters,lamda[0])
    theta1_5, J_history_5 = gradientDescentMulti2(new_train_array_norm, new_prices, theta, alpha2, num_iters,lamda[1])
    theta1_6, J_history_6 = gradientDescentMulti2(new_train_array_norm, new_prices, theta, alpha2, num_iters,lamda[2])

    hypo2_thetas=np.array([theta1_4,theta1_5,theta1_6])
    cv_4=computeCostcrossV2(new_cross_validation_norm,new_prices,theta1_4)
    cv_5=computeCostcrossV2(new_cross_validation_norm,new_prices,theta1_5)
    cv_6=computeCostcrossV2(new_cross_validation_norm,new_prices,theta1_6)

    cv_array2=np.array([cv_4,cv_5,cv_6])
    best_lamda_2=lamda[np.argmin(cv_array2,axis=0)]
    best_theta_2=hypo2_thetas[np.argmin(cv_array2,axis=0)]
    cost_test1=computeCostMulti(new_testing_norm,new_prices,best_theta_1,best_lamda_1)

    theta1_7, J_history_7 = gradientDescentMulti3(new_train_array_norm, new_prices, theta, alpha3, num_iters,lamda[0])
    theta1_8, J_history_8 = gradientDescentMulti3(new_train_array_norm, new_prices, theta, alpha3, num_iters,lamda[1])
    theta1_9, J_history_9 = gradientDescentMulti3(new_train_array_norm, new_prices, theta, alpha3, num_iters,lamda[2])

    hypo3_thetas=np.array([theta1_7,theta1_8,theta1_9])
    cv_7=computeCostcrossV3(new_cross_validation_norm,new_prices,theta1_7)
    cv_8=computeCostcrossV3(new_cross_validation_norm,new_prices,theta1_8)
    cv_9=computeCostcrossV3(new_cross_validation_norm,new_prices,theta1_9)

    cv_array3=np.array([cv_7,cv_8,cv_9])
    best_lamda_3=lamda[np.argmin(cv_array3,axis=0)]
    best_theta_3=hypo3_thetas[np.argmin(cv_array3,axis=0)]
    cost_test1=computeCostMulti(new_testing_norm,new_prices,best_theta_1,best_lamda_1)
    print(cv_1)
    print(cost_test1)
    cost_test2=computeCostMulti2(new_testing_norm,new_prices,best_theta_2,best_lamda_2)
    print(cv_4)
    print(cost_test2)

    cost_test3=computeCostMulti3(new_testing_norm,new_prices,best_theta_3,best_lamda_3)
    print(cv_7)
    print(cost_test3)
    pyplot.figure()
    pyplot . plot ( np . arange ( len ( J_history_1 )), J_history_1 , lw = 2 , label='h1')
    pyplot.figure()
    pyplot . plot ( np . arange ( len ( J_history_4 )), J_history_4 , lw = 2, label='h2' )
    pyplot.figure()
    pyplot . plot ( np . arange ( len ( J_history_7 )), J_history_7 , lw = 2, label='h3' )
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Cost J')
    pyplot.show()
    