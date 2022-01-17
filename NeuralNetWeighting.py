# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:00:15 2020

@author: nicholas.park
"""

import sys
import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import timeit
from matplotlib import cm

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

seed = np.random.seed(5)

# Obtain dummy columns
mushdata = pd.read_csv("~\\Desktop\\CS7641\\Randomized Optimization\\mushrooms.csv")

dummycolumns = mushdata.columns[1:]
mushdata = pd.get_dummies(mushdata, columns=dummycolumns,drop_first = True)
mushdata = mushdata.rename(columns={'class': 'Poisonous'})

mushdata.loc[(mushdata.iloc[:,0]=="e"),"Poisonous"] = 0
mushdata.loc[(mushdata.iloc[:,0]=="p"),"Poisonous"] = 1


col = mushdata.columns.tolist()
col.append(col.pop(col.index("Poisonous")))

mushdata =mushdata[col]


mushdata = mushdata.sample(frac= 1,random_state = seed) #shuffle
mushdata = mushdata.astype(int)
mushdata = mushdata.dropna()


mushdatatrain = mushdata.iloc[:int(len(mushdata)*0.9)]
mushdatatest = mushdata.iloc[int(len(mushdata)*0.9):]

#print(len(mushdata.columns))





# def NN_param_selection(X, y, nfolds):
#     hyperparameters = {
#     # 'hidden_nodes': [[25],[20],[10],[20,10],[10,5],[20,5]],
#     'learning_rate': [ 0.001, 0.01, 0.1],}
#     # 'learning_rate': ['constant','adaptive', 'invscaling'],}
#     grid_search = GridSearchCV(mlrose_hiive.NeuralNetwork(hidden_nodes = [25], algorithm = 'gradient_descent', activation = 'sigmoid', max_attempts = 200, learning_rate = 0.001), hyperparameters, cv=nfolds)#Fit the model
#     grid_search.fit(X,y)
#     grid_search.best_params_
    
#     return grid_search.best_params_

#{'activation': 'identity', 'hidden_nodes': [20, 10], 'learning_rate': 0.001, 'max_attempts': 200}


X, y = mushdatatrain.iloc[:,:-1], mushdatatrain.iloc[:,-1] 
X_test, y_test = mushdatatest.iloc[:,:-1], mushdatatest.iloc[:,-1] 
#print(NN_param_selection(X,y, nfolds =5) ) #{'hidden_nodes': [25], 'learning_rate': 0.001}

# rhc_hyperparameters = {
#   'restarts': [10],#, 200, 400, 800, 1600],
#   'learning_rate': [0.001]}#,0.01,0.1]}
# sa_hyperparameters = {
#   'schedule': [mlrose_hiive.ExpDecay(5,0.001),mlrose_hiive.ExpDecay(10,0.001),mlrose_hiive.ExpDecay(1,0.001), 
#   mlrose_hiive.ExpDecay(10,0.01), mlrose_hiive.ExpDecay(5,0.01), mlrose_hiive.ExpDecay(1,0.01),
#   mlrose_hiive.ExpDecay(10,0.0001), mlrose_hiive.ExpDecay(5,0.0001), mlrose_hiive.ExpDecay(1,0.0001),
#   mlrose_hiive.GeomDecay(10,0.9999),  mlrose_hiive.GeomDecay(10,0.999),  mlrose_hiive.GeomDecay(10,0.99),
#     mlrose_hiive.GeomDecay(5,0.9999),   mlrose_hiive.GeomDecay(5,0.999),  mlrose_hiive.GeomDecay(5,0.99),
#     mlrose_hiive.GeomDecay(1,0.9999),  mlrose_hiive.GeomDecay(1,0.999),  mlrose_hiive.GeomDecay(1,0.99)],
#     'learning_rate': [0.001,0.01,0.1]}
# ga_hyperparameters = {
#   'pop_size': [100, 200, 400],
#   'learning_rate': [0.001,0.01,0.1],
#   'mutation_prob': [0.05,0.1,0.2]}


# gs_rhc = GridSearchCV(mlrose_hiive.NeuralNetwork(hidden_nodes = [25], early_stopping =True, algorithm = 'random_hill_climb', activation = 'sigmoid', max_attempts = 300), rhc_hyperparameters, cv =5)
# gs_rhc.fit(X,y)
# print(gs_rhc.best_params_) # {'restarts' : 200, 'learning_rate': 0.001}



# gs_sa = GridSearchCV(mlrose_hiive.NeuralNetwork(hidden_nodes = [25], early_stopping =True,algorithm = 'simulated_annealing', activation = 'sigmoid', max_attempts = 300), sa_hyperparameters, cv =5)
# gs_sa.fit(X,y)
# print(gs_sa.best_params_) # {'schedule' :  mlrose_hiive.GeomDecay(5,0.99), 'learning_rate': 0.001}

# gs_ga = GridSearchCV(mlrose_hiive.NeuralNetwork(hidden_nodes = [25], early_stopping =True,algorithm = 'genetic_alg', activation = 'sigmoid', max_attempts = 300), ga_hyperparameters, cv =5)
# gs_ga.fit(X,y)
# print(gs_ga.best_params_) # {'pop_size' : 200, 'learning_rate' : 0.001, 'mutation_prob' : 0.01}

fc_gd = []
fc_rhc = []
fc_sa = []
fc_ga = []

rt_gd = []
rt_rhc = []
rt_sa = []
rt_ga = []

y_train_accuracy = []
y_test_accuracy = []
gd_test_times = []


for i in range(7):
        
    y_train_accuracy = []
    y_test_accuracy = []
    

    seed = np.random.seed(i*10)
    starttime = timeit.default_timer()

    GD_NN = mlrose_hiive.NeuralNetwork(hidden_nodes = [25], activation = 'sigmoid',\
                                  algorithm = 'gradient_descent', max_attempts = 200,  max_iters = 1000,  \
                                  bias = True, is_classifier = True, random_state = seed ,learning_rate = 0.1, \
                                  early_stopping = True,  curve = True)
    GD_NN.fit(X,y)
    endtime = timeit.default_timer()
    gd_test_times.append(endtime-starttime)


    fc_gd.append(GD_NN.fitness_curve)
    rt_gd.append(GD_NN.runtime)
    
    y_train_accuracy.append(accuracy_score(GD_NN.predict(X), y))
    y_test_accuracy.append(accuracy_score(GD_NN.predict(X_test), y_test))

fc_gd_mean, fc_gd_std = tolerant_mean(fc_gd)
rt_gd_mean, rt_gd_std = tolerant_mean(rt_gd)  


print("Gradient Descent: mean train accuracy {:f}".format(np.mean(y_train_accuracy)))
print("Gradient Descent: mean test accuracy {:f}".format(np.mean(y_test_accuracy)))


print("-----------------------")


y_train_accuracy = []
y_test_accuracy = []
test_times = []

for i in range(7):
    

    seed = np.random.seed(i*10)

    starttime = timeit.default_timer()
    RHC_NN = mlrose_hiive.NeuralNetwork(hidden_nodes = [25], activation = 'sigmoid',\
                                  algorithm = 'random_hill_climb', max_attempts = 200,  restarts=20 , max_iters = 1000,  \
                                  bias = True, is_classifier = True, random_state = seed ,learning_rate = 20, \
                                  early_stopping = True,  curve = True)
    RHC_NN.fit(X,y)
    endtime = timeit.default_timer()

    fc_rhc.append(RHC_NN.fitness_curve)
    rt_rhc.append(RHC_NN.runtime)
    test_times.append(endtime-starttime)
    
    y_train_accuracy.append(accuracy_score(RHC_NN.predict(X), y))
    y_test_accuracy.append(accuracy_score(RHC_NN.predict(X_test), y_test))

fc_rhc_mean, fc_rhc_std = tolerant_mean(fc_rhc)
rt_rhc_mean, rt_rhc_std = tolerant_mean(rt_rhc)  


print("Randomized Hill Climbing: mean train accuracy {:f}".format(np.mean(y_train_accuracy)))
print("Randomized Hill Climbing: mean test accuracy {:f}".format(np.mean(y_test_accuracy)))
print("-----------------------")

y_train_accuracy = []
y_test_accuracy = []


for i in range(10):
       
 
    seed = np.random.seed(i*10)

    SA_NN = mlrose_hiive.NeuralNetwork(hidden_nodes = [25], activation = 'sigmoid',\
                                  algorithm = 'simulated_annealing', max_attempts = 200, max_iters = 5000,  \
                                  bias = True, is_classifier = True, schedule = mlrose_hiive.GeomDecay(5,0.99), random_state = seed ,learning_rate = 40 ,\
                                  early_stopping = True,  curve = True)
    SA_NN.fit(X,y)
    fc_sa.append(SA_NN.fitness_curve)
    rt_sa.append(SA_NN.runtime)
    
    y_train_accuracy.append(accuracy_score(SA_NN.predict(X), y))
    y_test_accuracy.append(accuracy_score(SA_NN.predict(X_test), y_test))

fc_sa_mean, fc_sa_std = tolerant_mean(fc_sa)
rt_sa_mean, rt_sa_std = tolerant_mean(rt_sa)  

print("Simulated Annealing: mean train accuracy {:f}".format(np.mean(y_train_accuracy)))
print("Simulated Annealing: mean test accuracy {:f}".format(np.mean(y_test_accuracy)))



print("-----------------------")

y_train_accuracy = []
y_test_accuracy = []

for i in range(5):
        
    y_train_accuracy = []
    y_test_accuracy = []
    

    seed = np.random.seed(i*10)

    GA_NN = mlrose_hiive.NeuralNetwork(hidden_nodes = [25], activation = 'sigmoid',\
                                  algorithm = 'genetic_alg', max_attempts = 200,  max_iters = 1000,  \
                                  bias = True, pop_size= 100, mutation_prob = 0.1, is_classifier = True, random_state = seed ,learning_rate = 1, \
                                  early_stopping = True,  curve = True)
    GA_NN.fit(X,y)
    fc_ga.append(GA_NN.fitness_curve)
    rt_ga.append(GA_NN.runtime)
    
    y_train_accuracy.append(accuracy_score(GA_NN.predict(X), y))
    y_test_accuracy.append(accuracy_score(GA_NN.predict(X_test), y_test))

fc_ga_mean, fc_ga_std = tolerant_mean(fc_ga)
rt_ga_mean, rt_ga_std = tolerant_mean(rt_ga)  

print("Genetic Algorithm: mean train accuracy {:f}".format(np.mean(y_train_accuracy)))
print("Genetic Algorithm: mean test accuracy {:f}".format(np.mean(y_test_accuracy)))







plt.figure(0, figsize=(13, 4))
plt.subplot(1,2,1)
plt.title("Gradient Descent and RHC Fitness Curve")
plt.xlabel('Iteration')

plt.plot(np.arange(1,len(fc_gd_mean)+1),fc_gd_mean,label="Gradient Descent",linewidth = 1)
plt.fill_between(np.arange(1,len(fc_gd_mean)+1), (fc_gd_mean - fc_gd_std), (fc_gd_mean + fc_gd_std), alpha=.1)
plt.plot(np.arange(1,len(fc_rhc_mean)+1),fc_rhc_mean,label="Randomized Hill Climbing",linewidth = 1)
plt.fill_between(np.arange(1,len(fc_rhc_mean)+1), (fc_rhc_mean - fc_rhc_std), (fc_rhc_mean + fc_rhc_std), alpha=.1)

plt.ylabel('Fitness Function Value')
plt.grid(linewidth = 0.2)    
plt.legend(loc=0,fontsize="small")
plt.xlim(-30,1000)

plt.subplot(1,2,2)
plt.title("Cumulative Runtime vs iteration")
plt.xlabel('Iteration')

plt.plot(np.arange(1,len(rt_gd_mean)+1),rt_gd_mean,label="Gradient Descent",linewidth = 1)
plt.fill_between(np.arange(1,len(rt_gd_mean)+1), (rt_gd_mean - rt_gd_std), (rt_gd_mean + rt_gd_std), alpha=.1)
plt.plot(np.arange(1,len(rt_rhc_mean)+1),rt_rhc_mean,label="Randomized Hill Climbing",linewidth = 1)
plt.fill_between(np.arange(1,len(rt_rhc_mean)+1), (rt_rhc_mean - rt_rhc_std), (rt_rhc_mean + rt_rhc_std), alpha=.1)

plt.ylabel('Runtime in seconds')
plt.grid(linewidth = 0.2)    
plt.legend(loc=0,fontsize="small")
plt.xscale("log")
plt.xlim(1,1000)


plt.savefig("Algorithm Fitness and Runtime Curve Neural Net 1.png")







plt.figure(1, figsize=(13, 4))
plt.subplot(1,2,1)
plt.title("Gradient Descent and Simulated Annealing Fitness Curve")
plt.xlabel('Iteration')

plt.plot(np.arange(1,len(fc_gd_mean)+1),fc_gd_mean,label="Gradient Descent",linewidth = 1)
plt.fill_between(np.arange(1,len(fc_gd_mean)+1), (fc_gd_mean - fc_gd_std), (fc_gd_mean + fc_gd_std), alpha=.1)
plt.plot(np.arange(1,len(fc_sa_mean)+1),fc_sa_mean,label="Simulated Annealing",linewidth = 1)
plt.fill_between(np.arange(1,len(fc_sa_mean)+1), (fc_sa_mean - fc_sa_std), (fc_sa_mean + fc_sa_std), alpha=.1)


plt.ylabel('Fitness Function Value')
plt.grid(linewidth = 0.2)    
plt.legend(loc=0,fontsize="small")
plt.xlim(-30,1000)

plt.subplot(1,2,2)
plt.title("Cumulative Runtime vs iteration")
plt.xlabel('Iteration')

plt.plot(np.arange(1,len(rt_gd_mean)+1),rt_gd_mean,label="Gradient Descent",linewidth = 1)
plt.fill_between(np.arange(1,len(rt_gd_mean)+1), (rt_gd_mean - rt_gd_std), (rt_gd_mean + rt_gd_std), alpha=.1)
plt.plot(np.arange(1,len(rt_sa_mean)+1),rt_sa_mean,label="Simulated Annealing",linewidth = 1)
plt.fill_between(np.arange(1,len(rt_sa_mean)+1), (rt_sa_mean - rt_sa_std), (rt_sa_mean + rt_sa_std), alpha=.1)


plt.ylabel('Runtime in seconds')
plt.grid(linewidth = 0.2)    
plt.legend(loc=0,fontsize="small")
plt.xscale("log")
plt.xlim(1,1000)


plt.savefig("Algorithm Fitness and Runtime Curve Neural Net 2.png")





plt.figure(2, figsize=(13, 4))
plt.subplot(1,2,1)
plt.title("Gradient Descent and Genetic Algorithm Fitness Curve")
plt.xlabel('Iteration')

plt.plot(np.arange(1,len(fc_gd_mean)+1),fc_gd_mean,label="Gradient Descent",linewidth = 1)
plt.fill_between(np.arange(1,len(fc_gd_mean)+1), (fc_gd_mean - fc_gd_std), (fc_gd_mean + fc_gd_std), alpha=.1)
plt.plot(np.arange(1,len(fc_ga_mean)+1),fc_ga_mean,label="Genetic Algorithm",linewidth = 1)
plt.fill_between(np.arange(1,len(fc_ga_mean)+1), (fc_ga_mean - fc_ga_std), (fc_ga_mean + fc_ga_std), alpha=.1)


plt.ylabel('Fitness Function Value')
plt.grid(linewidth = 0.2)    
plt.legend(loc=0,fontsize="small")
plt.xlim(-30,1000)

plt.subplot(1,2,2)
plt.title("Cumulative Runtime vs iteration")
plt.xlabel('Iteration')

plt.plot(np.arange(1,len(rt_gd_mean)+1),rt_gd_mean,label="Gradient Descent",linewidth = 1)
plt.fill_between(np.arange(1,len(rt_gd_mean)+1), (rt_gd_mean - rt_gd_std), (rt_gd_mean + rt_gd_std), alpha=.1)
plt.plot(np.arange(1,len(rt_ga_mean)+1),rt_ga_mean,label="Genetic Algorithm",linewidth = 1)
plt.fill_between(np.arange(1,len(rt_ga_mean)+1), (rt_ga_mean - rt_ga_std), (rt_ga_mean + rt_ga_std), alpha=.1)


plt.ylabel('Runtime in seconds')
plt.grid(linewidth = 0.2)    
plt.legend(loc=0,fontsize="small")
plt.xscale("log")
plt.xlim(1,1000)


plt.savefig("Algorithm Fitness and Runtime Curve Neural Net 3.png")







plt.figure(3, figsize=(13, 4))
plt.subplot(1,2,1)
plt.title("Average Time to reach convergence")
plt.xlabel('time in seconds')

plt.plot(rt_gd_mean,fc_gd_mean,label="Gradient Descent",linewidth = 1)
plt.fill_between(rt_gd_mean, (fc_gd_mean - fc_gd_std), (fc_gd_mean + fc_gd_std), alpha=.1)
plt.plot(rt_rhc_mean,fc_rhc_mean,label="Randomized Hill Climbing",linewidth = 1)
plt.fill_between(rt_rhc_mean, (fc_rhc_mean - fc_rhc_std), (fc_rhc_mean + fc_rhc_std), alpha=.1)
plt.plot(rt_sa_mean,fc_sa_mean,label="Simulated Annealing",linewidth = 1)
plt.fill_between(rt_sa_mean, (fc_sa_mean - fc_sa_std), (fc_sa_mean + fc_sa_std), alpha=.1)
plt.plot(rt_ga_mean,fc_ga_mean,label="Genetic Algorithm",linewidth = 1)
plt.fill_between(rt_ga_mean, (fc_ga_mean - fc_ga_std), (fc_ga_mean + fc_ga_std), alpha=.1)

plt.ylabel('Fitness Function Value')
plt.grid(linewidth = 0.2)    
plt.legend(loc=0,fontsize="small")
plt.ylim(-1,20)
plt.xlim(0,150)


plt.subplot(1,2,2)

plt.title("Average Cumulative time over iteration")
plt.xlabel('Iteration')

plt.plot(np.arange(1,len(rt_gd_mean)+1),rt_gd_mean,label="Gradient Descent",linewidth = 1)
plt.fill_between(np.arange(1,len(rt_gd_mean)+1), (rt_gd_mean - rt_gd_std), (rt_gd_mean + rt_gd_std), alpha=.1)
plt.plot(np.arange(1,len(rt_rhc_mean)+1),rt_rhc_mean,label="Randomized Hill Climbing",linewidth = 1)
plt.fill_between(np.arange(1,len(rt_rhc_mean)+1), (rt_rhc_mean - rt_rhc_std), (rt_rhc_mean + rt_rhc_std), alpha=.1)
plt.plot(np.arange(1,len(rt_sa_mean)+1),rt_sa_mean,label="Simulated Annealing",linewidth = 1)
plt.fill_between(np.arange(1,len(rt_sa_mean)+1), (rt_sa_mean - rt_sa_std), (rt_sa_mean + rt_sa_std), alpha=.1)
plt.plot(np.arange(1,len(rt_ga_mean)+1),rt_ga_mean,label="Genetic Algorithm",linewidth = 1)
plt.fill_between(np.arange(1,len(rt_ga_mean)+1), (rt_ga_mean - rt_ga_std), (rt_ga_mean + rt_ga_std), alpha=.1)


plt.ylabel('Time in seconds')
plt.grid(linewidth = 0.2)    
plt.legend(loc=0,fontsize="small")
plt.ylim(-1,18)

plt.xlim(1,1000)
plt.savefig("NN Algorithms Convergence.png")






