# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:19:46 2020

@author: nicholas.park
"""
import sys
import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit


#seed = np.random.seed(2)
#seed = np.random.seed(1)
seed = np.random.seed()

fitness = mlrose_hiive.ContinuousPeaks()
problem = mlrose_hiive.DiscreteOpt(length=30, fitness_fn=fitness, maximize=True, max_val=2)

algorithms_to_run = []

if 'RHC' in algorithms_to_run:
    # RHC
    print("Running RHC...")
    max_attempts = [500]
    restarts = [0, 200,400,800,1600]

    plt.figure(0,figsize=(7, 5))
    plt.title("RHC Parameter Tuning Fitness Curve")
    plt.xlabel('Iteration')
    plt.xlim(-10,600)
    plt.ylabel('Fitness Function Value')
    plt.grid(linewidth = 0.2)

    for i in max_attempts:
        for j in restarts:
            starttime=timeit.default_timer()
            bs,bf,fc,rt = mlrose_hiive.random_hill_climb(problem=problem, max_attempts=i, restarts = j,curve=True,random_state=seed)#,state_fitness_callback=eval_times, callback_user_info=[3])
            endtime = timeit.default_timer()
            plt.plot(np.arange(1,len(fc)+1),fc,label="ma = {}, r = {}, in {:.2f}s".format(i,j, endtime - starttime))
            output = dict(
                         best_state=bs,
                         best_function=bf,
                         fitness_curve=fc,
                         runtime=rt
                         )
            np.save('RHC_output_{:n}_restarts={:d}.npy'.format(i,j), output)
    plt.legend(loc="lower right",fontsize="small")

    plt.savefig("RHC Parameter Tuning Fitness Curve CPeaks.png")

if 'SA' in algorithms_to_run:
    # SA
    print("Running SA...") 
    decay_arr = ['ArithDecay', 'ExpDecay', 'GeomDecay']
    init_temp_arr = [10]#5, 10]
    second_arg_arr = [[0.0001,0.001,0.01], [0.005, 0.01 , 0.02], [0.999,0.99,0.9]]
    linestyles_dict = dict(
                          ArithDecay='solid',
                          ExpDecay='--',
                          GeomDecay='dotted'
                          )
    plt.figure(1,figsize=(7.5,5))
    from matplotlib import cm

    colors_list = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'magenta', 'k']
    for dx, decay in enumerate(decay_arr):
        i = 0
        for init_temp in init_temp_arr:
            for second_arg in second_arg_arr[dx]:
                schedule_obj = getattr(mlrose_hiive, decay)(init_temp, second_arg)
                if decay in ['ArithDecay', 'GeomDecay']:
                    schedule_label = '{:s}, init_temp={:d}, decay={:f}'.format(decay, init_temp, second_arg)
                else: # ExpDecay
                    schedule_label = '{:s}, init_temp={:d}, exp_const={:f}'.format(decay, init_temp, second_arg)
                starttime =timeit.default_timer()
                bs,bf,fc,rt = mlrose_hiive.simulated_annealing(problem=problem, max_attempts=1000, schedule=schedule_obj, curve=True, random_state=seed)#,state_fitness_callback=eval_times, callback_user_info=[3])
                endtime = timeit.default_timer()
                plt.plot(np.arange(1, len(fc)+1), fc, label="s = {}, in {:.2f}s".format(schedule_label, endtime - starttime), linestyle=linestyles_dict[decay], linewidth = 1, color=cm.hsv(i))
                i += 1.0/(len(init_temp_arr)*len(second_arg_arr))   

                output = dict(
                 best_state=bs,
                 best_function=bf,
                 fitness_curve=fc,
                 runtime=rt
                 )
                np.save('SA_output_{:s}_init_temp={:d}_decay={:f}.npy'.format(decay, init_temp, second_arg), output)
    plt.xlim(-50,3500)
    plt.title("SA Parameter Tuning Fitness Curve, init_temp = 10")
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Function Value')

    plt.grid(linewidth = 0.2)   
    plt.legend(loc="lower right",fontsize="small")
    plt.savefig("SA Parameter Tuning Fitness Curve CPeaks.png")

# # GA
if 'GA' in algorithms_to_run:
    print("Running GA...")
    #max_attempts = [10000]
    pop_size  = [500,]# 500, 1000]
    pop_breed_percent = [0.5, 0.75, 1]
    mutation_prob = [0.1, 0.3, 0.5, 0.7]
                 
    plt.figure(2, figsize=(7, 5))
    from matplotlib import cm

    linestyles_list = ['solid','dotted','dashed']
    colors_list = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'magenta',]
    for i in pop_size:
        colorindex = 0

        for j in pop_breed_percent:
            for k in mutation_prob:
                print("Running pop_size={:d}, pop_breed_percent={:f}, mutation_prob={:f}...".format(i, j, k))
                starttime = timeit.default_timer()

                bs, bf, fc, rt = mlrose_hiive.genetic_alg(problem=problem, max_attempts=200, pop_size = i, pop_breed_percent= j, mutation_prob=k,curve=True, random_state=seed)#,state_fitness_callback=eval_times, callback_user_info=[3])
                endtime = timeit.default_timer()
                plt.plot(np.arange(1,len(fc)+1), fc, label="ps = {}, pbp = {}, mp = {}, in {:.2f}s".format(i,j,k, endtime - starttime), color = cm.hsv(colorindex))      
                colorindex+= 1.0/(len(pop_breed_percent)*len(mutation_prob))
                output = dict(
                             best_state=bs,
                             best_function=bf,
                             fitness_curve=fc,
                             runtime=rt
                             )
                np.save("GA_output_pop_size={:d}_pop_breed_percent={:f}_mutation_prob={:f}.npy".format(i, j, k), output)
    plt.title("GA Parameter Tuning Fitness Curve, pop_size = 500")
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Function Value')
    plt.grid(linewidth = 0.2)

    plt.legend(loc="lower right",fontsize="small")
    plt.savefig("GA Parameter Tuning Fitness Curve CPeaks.png")


# MIMIC
if 'MIMIC' in algorithms_to_run:
    print("Running MIMIC...")
    pop_size  = [100, 500,1000]
    keep_pct  = [0.1, 0.3, 0.5, 0.7]
    from matplotlib import cm

    plt.figure(3, figsize=(7, 5))
    colorindex = 0
    for i in pop_size:
        for j in keep_pct:
                print("Running pop_size={:d}, keep_pct={:f}".format(i, j))
                starttime = timeit.default_timer()
                bs,bf,fc,rt = mlrose_hiive.mimic(problem=problem, max_attempts=300, pop_size = i, keep_pct= j, curve=True,random_state=seed,fast_mimic=True)#,state_fitness_callback=eval_times, callback_user_info=[3])
                endtime = timeit.default_timer()
                plt.plot(np.arange(1,len(fc)+1),fc,label="ps = {}, pct = {}, in {:.2f}s".format(i,j,endtime - starttime),linewidth = 1, color =cm.hsv(colorindex))  
                colorindex+= 1.0/(len(pop_size)*len(keep_pct))
                output = dict(
                             best_state=bs,
                             best_function=bf,
                             fitness_curve=fc,
                             runtime=rt
                             )
                np.save("MIMIC_output_pop_size={:d}, keep_pct={:f}.npy".format(i, j), output)
    plt.title("MIMIC Parameter Tuning Fitness Curve")
    plt.xlim(-10,300)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Function Value')
    plt.grid(linewidth = 0.2)    
    plt.legend(loc="lower right",fontsize="small")

    plt.savefig("MIMIC Parameter Tuning Fitness Curve CPeaks.png")


from matplotlib import cm

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)



fc_rhc = []
fc_sa = []
fc_ga = []
fc_mimic = []

for i in range(5):
    seed = np.random.seed(i*10)
    bs,bf,fc, rt_rhc = mlrose_hiive.random_hill_climb(problem=problem, max_attempts = 500, restarts = 400,curve=True,random_state=seed)#,state_fitness_callback=eval_times, callback_user_info=[3])
    fc_rhc.append(fc)
fc_rhc = tolerant_mean(fc_rhc)


for i in range(5):
    seed = np.random.seed(i*10)
    bs,bf,fc, rt_sa = mlrose_hiive.simulated_annealing(problem=problem, max_attempts = 500,schedule=mlrose_hiive.GeomDecay(5,0.9), curve=True, random_state=seed)#,state_fitness_callback=eval_times, callback_user_info=[3])
    fc_sa.append(fc)
fc_sa = tolerant_mean(fc_sa)


for i in range(5):
    seed = np.random.seed(i*10)
    bs, bf, fc, rt_ga = mlrose_hiive.genetic_alg(problem=problem, max_attempts = 500, pop_size = 100, pop_breed_percent= 0.75, mutation_prob=0.7,curve=True, random_state=seed)#,state_fitness_callback=eval_times, callback_user_info=[3])
    fc_ga.append(fc)
fc_ga = tolerant_mean(fc_ga)

for i in range(5):
    seed = np.random.seed(i*10)
    bs,bf, fc, rt_mimic = mlrose_hiive.mimic(problem=problem, max_attempts = 500,pop_size = 500, keep_pct= 0.7, curve=True,random_state=seed,fast_mimic=True)#,state_fitness_callback=eval_times, callback_user_info=[3])
    fc_mimic.append(fc)
fc_mimic = tolerant_mean(fc_mimic)



plt.figure(4, figsize=(13, 5.5))
plt.subplot(1,2,1)
plt.title("Continuous Peaks: Algorithm Comparisons Fitness Curve")
plt.xlabel('Iteration')
plt.plot(np.arange(1,len(fc_rhc[0])+1),fc_rhc[0],label="RHC: r = 400",linewidth = 1)
plt.fill_between(np.arange(1,len(fc_rhc[0])+1), (fc_rhc[0]-fc_rhc[1]), (fc_rhc[0] +fc_rhc[1]), alpha=.1)
plt.plot(np.arange(1, len(fc_sa[0])+1), fc_sa[0], label="SA: s = GeomDecay, rate = 0.9, init_temp = 5", linewidth =1)
plt.fill_between(np.arange(1,len(fc_sa[0])+1), (fc_sa[0]-fc_sa[1]), (fc_sa[0] +fc_sa[1]), alpha=.1)
plt.plot(np.arange(1,len(fc_ga[0])+1), fc_ga[0], label="GA: ps = 100, pbp = 0.75, mp = 0.7", linewidth = 1)      
plt.fill_between(np.arange(1,len(fc_ga[0])+1), (fc_ga[0]-fc_ga[1]), (fc_ga[0] +fc_ga[1]), alpha=.1)
plt.plot(np.arange(1,len(fc_mimic[0])+1),fc_mimic[0],label="MIMIC: ps = 500, pct = 0.7", linewidth = 1)  
plt.fill_between(np.arange(1,len(fc_mimic[0])+1), (fc_mimic[0]-fc_mimic[1]), (fc_mimic[0] +fc_mimic[1]), alpha=.1)
plt.ylabel('Fitness Function Value')
plt.grid(linewidth = 0.2)    
plt.legend(loc=0,fontsize="small")
plt.xlim(-20,600)

plt.subplot(1,2,2)
plt.title("Continuous Peaks: Algorithm Comparisons Runtime Curve")
plt.xlabel('Iteration')

plt.plot(np.arange(1,len(rt_rhc)+1),rt_rhc,label="RHC: r = 400",linewidth = 1)
plt.plot(np.arange(1, len(rt_sa)+1), rt_sa, label="SA: s = GeomDecay, rate = 0.9, init_temp = 5", linewidth = 1)
plt.plot(np.arange(1,len(rt_ga)+1), rt_ga, label="GA: ps = 100, pbp = 0.75, mp = 0.7", linewidth = 1)      
plt.plot(np.arange(1,len(rt_mimic)+1),rt_mimic,label="MIMIC: ps = 500, pct = 0.7", linewidth = 1)  

plt.ylabel('Runtime in seconds')
plt.grid(linewidth = 0.2)    
plt.legend(loc=0,fontsize="small")
plt.xlim(-20,600)


plt.savefig("Algorithm Fitness and Runtime Curve CPeaks.png")


