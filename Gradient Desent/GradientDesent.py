# -*- coding: utf-8 -*-
#coding=utf-8
import numpy as np

#批量梯度下降法（BGD）
def BatchGradientDescent(X,y,theta,m,n_itrations,alpha):
    for iteration in range(n_itrations):
        gradients = 1/m*X.T.dot(X.dot(theta)-y)       
        theta = theta - alpha*gradients
    return theta
#随机梯度下降法（SGD）
def StochasticGradientDescent(X,y,theta,m,n_itrations,alpha):
    for iteration in range(n_itrations):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = xi.T.dot(xi.dot(theta)-yi)
        theta = theta - alpha*gradients
    return theta
        
X = 2*np.random.rand(100,1)
y = 4+3*X+np.random.randn(100,1)

X_b = np.c_[np.ones((100,1)),X]#增加X0 = 1
theta = np.random.randn(2,1)#随机初始theta值

theta_best_1 = BatchGradientDescent(X_b,y,theta,100,100,0.1)
theta_best_2 = StochasticGradientDescent(X_b,y,theta,100,100,0.1)

