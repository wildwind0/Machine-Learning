# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 14:25:43 2017
EM算法求解高斯混合模型参数
@author: wildwind_
"""

import numpy as np
gamma0 = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]).reshape(1,15)
gamma = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75,
                  -67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]).reshape(1,30)
alpha0 = 0.5
mean0 = [30, -30]
variance0 = [500, 500]

i = 1  #迭代次数

def calculate_norm(y, mean, variance):
    '''
    计算正态分布概率
    '''
    return (np.sqrt(2*np.pi*variance)**-1)*np.e**(-((y-mean)**2)/(2*variance))

def update_gamma(gamma, alpha, mean, variance):
    '''
    EM算法E步
    '''
    new_gamma_k1 = []
    new_gamma_k2 = []
    for y in gamma:
        norm_k1 = calculate_norm(y, mean[0], variance[0])
        norm_k2 = calculate_norm(y, mean[1], variance[1])
        
        new_gamma_k1.append((alpha*norm_k1)/(alpha*norm_k1+(1-alpha)*norm_k2))
        new_gamma_k2.append(((1-alpha)*norm_k2)/(alpha*norm_k1+(1-alpha)*norm_k2))
    return np.append(np.array(new_gamma_k1), np.array(new_gamma_k2)).reshape(1,30)
        
def update_parameter(gamma, mean):
    '''
    EM算法M步
    '''
    new_mean_k1 = (np.dot(gamma0, gamma[0][0:15].reshape(1,15).T))/gamma[0][0:15].sum()    
    new_variance_k1 = (np.dot((gamma0 - mean[0])**2,gamma[0][0:15].T))/gamma[0][0:15].sum()
    
    new_mean_k2 = (np.dot(gamma0, gamma[0][15:].reshape(1,15).T))/gamma[0][15:].sum()    
    new_variance_k2 = (np.dot((gamma0 - mean[1])**2,gamma[0][15:].T))/gamma[0][15:].sum()
    
    new_alpha = gamma[0][0:15].sum()/15
    
    return new_alpha, [new_mean_k1, new_mean_k2], [new_variance_k1, new_variance_k2]

def EM(gamma, alpha, mean, variance):
    global i 
    new_gamma = update_gamma(gamma0, alpha, mean, variance)
    
    new_alpha, new_mean, new_variance =  update_parameter(new_gamma, mean)
    difference = abs(new_alpha - alpha) + \
                 np.absolute((np.array(new_mean) - np.array(mean))).sum()+ \
                 np.absolute((np.array(new_variance) - np.array(variance))).sum() 

    if difference < np.e**-5 :  
        print('第 %d 次迭代结果：' % i)
        print('gamma1: ', new_gamma[0][0:15])
        print('gamma2: ', new_gamma[0][15:])
        print('alpha: ',new_alpha)
        print('mean: ', new_mean)
        print('variance: ', new_variance)
        return
    else:
        print('第 %d 次迭代结果：' % i)
        i += 1
        print('gamma1: ', new_gamma[0][0:15])
        print('gamma2: ', new_gamma[0][15:])
        print('alpha: ',new_alpha)
        print('mean: ', new_mean)
        print('variance: ', new_variance)
        return EM(new_gamma, new_alpha, new_mean, new_variance)
    
EM(gamma, alpha0, mean0, variance0)
