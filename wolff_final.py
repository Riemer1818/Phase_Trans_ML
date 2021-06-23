#%% modules

import numpy as np
from math import floor
from math import exp
from random import random

from copy import deepcopy as dc

import matplotlib.pyplot as plt

from scipy.io import savemat

from datetime import datetime

#%% initialization

# wether to save the file (1) or not (0)
# usefull for testing
# file directory at the bottom of the code
save_to_mat=1

# J value should be J>0; wolff doesn't work for antiferromagnet (I think)
J = 1

# enable show_plot to plot the last snapshot
# set temps and snapshots to 1 and only 1 dim in dimensions to do a single wolff iteration
# will compute for temp_min
show_plot=0

# spatial dimension spdim (currently not generalized to higer dims)
# list of matrix dimensions
spdim = 2
dimensions=[60]

# temperature range, number of temperatures, number of snapshots (simulations), average flips per site
temp_min=1.9
temp_max=2.9
temps=20
snapshots=5000
flips_per_site=2

# create neighbor list depending on spdim
dimlist=[]
for i in range(0,2*spdim):
    dimlist.append(np.zeros(spdim,dtype=int))
    dimlist[i][floor(i/2)]=(-1)**i
    dimlist[i]=tuple(dimlist[i])

# create list to store execution time for each dimension (all temperatures)
time_list=[]

# empty list that will hold the first cluster
cluster=[]

# %% full algorithm with wolff iterations

# execute algorithm
# for each dim in dimensions,
# for each temp temperatures between temp_min and temp_max
# do snapshots number of experiments
# with an average of flips_per_site number of flips per site
# start with a "pre-thermalized" state of all ones, this quickly resolves
# keeping the grids in between snapshots and temperatures (one continuous simulation)
# with increasing temperature
for dim in dimensions:
    
    # display the current dim
    # to get a sense of progress
    print(dim)
    
    # remember the number of sites
    number_of_sites=dim**spdim
    
    # size_tuple needed for creating an appropriately sized array of all ones
    temporary_list=[]
    for i in range(spdim):
        temporary_list.append(dim)
    size_tuple=tuple(temporary_list)

    
    # create dictionary of neighbors: (i,j) -> list of nbs of (i,j)  [UPGRADE TO HIGHER DIMS: not general]
    # also added truth value to the nbr to indicate wether it should be checked
    # more efficient for memory and keeps the nbr structure for removal
    dictionary_of_neighbours={}
    for i in range(dim):
        for j in range(dim):
            dictionary_of_neighbours[(i,j)]={}
            for move in dimlist:
                neighbour=tuple(map(lambda x, y: (x + y)%dim, (i,j), move))
                dictionary_of_neighbours[(i,j)][neighbour]=True
    
    
    # empty list to be filled with simulation results
    save_list=[]
    
    # matrix of ones for a thermalized head start
    M=np.ones(size_tuple)
    
    # start timer for current dim
    timer=datetime.now()
    
    # loop containing the wolff algorithm
    for temp in np.linspace(temp_min,temp_max,temps):
        
        # compute probability only once
        # depends only on J and temp
        adding_probability=1-exp(-2*J/temp)
        
        # start timer for current temp
        timer2=datetime.now()
        
        # halting variable such that flips_per_site points have been added to cluster
        desired_cluster_total=flips_per_site*dim**spdim
        
        # iterate over snapshots (individual experiments)
        for snap in range(snapshots):
            
            # variable to keep track of the total cluster sizes between iterations
            current_cluster_total=0
            
            # this is the actual wolff algorithm
            # dynamically decide the number of iterations depending on flips_per_site
            # efficiently handles cluster structure by keeping track of neighbours
            # main idea:
            # select a random point and add evenly aligned points to its cluster with some probability
            # see report for precise description
            while True:
                
                # reset neighbour dictionary
                # True means that a neigbour is eligible for the cluster selection
                # reset only the neighbours of points in the previous cluster
                for point in cluster:
                    for neighbour in dictionary_of_neighbours[point]:
                        dictionary_of_neighbours[neighbour][point]=True
                
                # empty list that will hold the current cluster
                cluster=[]
                
                # decide on a starting point
                start=tuple(np.random.randint(dim,size=spdim))
                
                # exclude the starting point from consideration in all of ITS neigbours
                # such that if one of those neighbours is selected in the algorithm
                # the starting point will not be considered in that iteration
                # the same idea is used in every iteration (that is, adding of a point to the cluster)
                for neighbour_of_start in dictionary_of_neighbours[start]:
                    dictionary_of_neighbours[neighbour_of_start][start]=False
                
                # add the starting point to the cluster
                cluster.append(start)
                
                # remember the cluster position (point of which neighbours are checked)
                # and the cluster length
                # to save computation time
                cluster_position=0
                cluster_len=1
                
                # execute main loop, adding equally aligned neighbours to a cluster according to some probability
                # check all added points only once and ignore neighbours already in the cluster
                # keep going untill all cluster points have been considered (cluster_len > cluster_position)
                while cluster_len > cluster_position:
                    
                    # set the centre point for the current check
                    # to be the one being pointed at by the variable cluster_position
                    centre_point=cluster[cluster_position]
                    
                    # check all the neighbours of the centre point
                    for neighbour_of_centre in dictionary_of_neighbours[centre_point]: # 2*spdim times
                        
                        # for every neighbour, check if all conditions are fufilled
                        # lazy execution: check conditions from left to right and quit checking at the first mismatch
                        # the lookup in M is probably fastest
                        # then the nested lookup in the neigbour dictionary
                        # if the value is False, it means that that neighbour is already in the cluster
                        # finally only generate a random number if both tests passed
                        if M[centre_point]==M[neighbour_of_centre] and dictionary_of_neighbours[centre_point][neighbour_of_centre] and adding_probability > random(): # lazy: quit if no match
                            
                            # if the neigbour has passed all tests
                            # add it to the cluster
                            # and keep track of the cluster length manually
                            cluster.append(neighbour_of_centre)
                            cluster_len=cluster_len+1
                            
                            # then, remove the selected neighbour from all of ITS neighbours
                            # just like has been done with the starting point start
                            for neigbour_of_the_neighbour_of_the_centre in dictionary_of_neighbours[neighbour_of_centre]:
                                dictionary_of_neighbours[neigbour_of_the_neighbour_of_the_centre][neighbour_of_centre]=False
                    
                    # move the current cluster pointer one up
                    # to indicate that the next point in the cluster should be the next centre point
                    # the while statement will terminate if this variable equals the length
                    # since it's a python index, it's one smaller than the length
                    # so it should terminate at equality
                    cluster_position=cluster_position+1
                    
                # flip all points in the cluster that has been formed (could be 1 point or entire grid)
                for point in cluster:
                    M[point]=-M[point]
                
                # halting condition if total points added to cluster exceeds flips_per_site
                if current_cluster_total<flips_per_site*number_of_sites:
                        current_cluster_total=current_cluster_total+cluster_len
                else:
                    break
            
            # for the current snapshot, save the final grid in a list
            save_list.append(dc(M))
        
        # for the current temperature, display the time spent on all of the computations
        print("%.3f" % temp,datetime.now()-timer2)
    
    #%% save to file or show plot
    
    # for the current dim
    # determine time spent on all computations (temp, snapshots, iterations)
    # and add it to a list
    # to be displayed at the end
    # save in seconds
    elapsed=datetime.now()-timer
    time_list.append((dim,elapsed.seconds))
    
    # show the current dim and it's total computation time
    print(dim,elapsed)
    
    # according to the parameters at the top of the code
    # show a plot of the final snapshot
    # or save the current list of grids to a (!!!) MATLAB (!!!) file
    # incorporate all parameters in fliename,
    # completely describing the experiment
    if show_plot==1:
        plt.imshow(M)
        plt.show()
    
    if save_to_mat==1:
        my_dictionary={'wolff':save_list}
        savemat(f"./final_w{dim}_temps{temps}_in{temp_min}to{temp_max}_snap{snapshots}_flips{flips_per_site}_{spdim}d_elapsed{elapsed.seconds}.mat",my_dictionary)
        #savemat("./outputmatlab.mat", my_dictionary)

# finally, show a list of all of the dims and their computation times (in whole seconds)
print(time_list)
