# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:14:38 2021

@author: jochem
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from functions.functions import *
        
if __name__ == "__main__":

    path_to_testdata = ""
    dir = "./data"    
    dirname = "normal_2D_20grid_10000itir_100step"
    path = os.path.join(dir, dirname)

    weights = ([np.load(os.path.join(path, 'weights.npy'), allow_pickle=True)])[0]
    bias    = ([np.load(os.path.join(path, 'bias.npy'), allow_pickle=True)])[0]

    dataset = np.load(path_to_testdata, allow_pickle=True)
    dataset = np.concatenate(dataset) # nu hebben we een lijst van shape (5000,2) dus 5000 lijsten van de vorm [temp, grid]

    number_of_training_data = 30 #what is this? 
    shape = [np.shape(i)[0] for i in weights]
    shape.append(np.shape(bias[-1])[0])

    nn = NeuralNetwork(shape, weights, bias, dataset, number_of_training_data) #eerste optie [200 ,50 , 30]
    
    nn.Desired_Out()
    foutmarge_training_data = []
    foutmarge_ongeziene_data = []
    weight_aanpas_groote = []
    nfactor_lijst = []


    T,y1,y1_std,y2,y2_std = nn.conclusieT()

    plt.figure()

    plt.errorbar(T,y1,y1_std)
    plt.errorbar(T,y2,y2_std)
