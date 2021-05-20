import matplotlib.pyplot as plt
import numpy as np

# K.L. Zwetsloot & B.Y. Cinemre

# =============================================================================
# This file can be used to generate visualizations of different lattices,
# for different parameter values.
# =============================================================================

# creates a list of the spin values of the neighbours using periodic boundary
# conditions.
def get_neighbors(grid,i,j):
    neighbor_list = []
    if i-1 < 0:
      neighbor_list.append(grid[s-1,j])
    else:
      neighbor_list.append(grid[i-1,j])
    if i+1 > s-1:
      neighbor_list.append(grid[0,j])
    else:
      neighbor_list.append(grid[i+1,j])
    if j-1 < 0:
      neighbor_list.append(grid[i,s-1])
    else:
      neighbor_list.append(grid[i,j-1])
    if j+1 > s-1:
      neighbor_list.append(grid[i,0])
    else:
      neighbor_list.append(grid[i,j+1])
    return neighbor_list

# function to calculate energy change of a spin flip
def delta_E(grid,i,j):
    delta_E = 2*J*grid[i,j]*sum(get_neighbors(grid,i,j))
    return delta_E

# metropolis algorithm, repeated N times
def metro_alg(grid,T_red,N):
    for i in range(N):
        x = np.random.randint(s)
        y = np.random.randint(s)
        DE = delta_E(grid,x,y)
        if DE <= 0:
            grid[x,y] = -grid[x,y]
        elif DE > 0:
            prob_flip = np.exp((-DE*J)/T_red)
            if np.random.random() < prob_flip:
                grid[x,y] = -grid[x,y]
    return grid

# parameter values
J = 1
T_red = 3
k = 1.0
s = 50
N = 10**4
size = (s,s)
randomize = True

# creates an initial grid configuration, either uniform or randomized
grid = np.ones((size), dtype=int)
if randomize == True:
    for i in range(s):
        for j in range (s):
            if np.random.random() < 0.5:
                grid[i,j] = -1

# iterates the metropolis algorithm to alter the grid and plots the final configuration
grid = metro_alg(grid, T_red, N)
plt.imshow(grid)
plt.axis("off")