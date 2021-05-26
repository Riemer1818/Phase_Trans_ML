# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:14:38 2021

@author: jochem
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from functions.functions import *
        
if __name__ == "__main__":

    dirname_train = sys.argv[1]
    dirname_test = sys.argv[2]

<<<<<<< HEAD
    dir_train = "./train_data"    
=======
    dir_train = "./data"    
>>>>>>> 422c5d3d4b23f12a4efb87ed1fd042a30f22de41
    path_train = os.path.join(dir_train, dirname_train)
    print("using: ", path_train, " as training inputfile")

    dir_test = "./test_data"
    path_test = os.path.join(dir_test, dirname_test)
    print("using: ", path_test, " as testing inputfile")

    weights = ([np.load(os.path.join(path_train, 'weights.npy'), allow_pickle=True)])[0]
    bias    = ([np.load(os.path.join(path_train, 'bias.npy'), allow_pickle=True)])[0]

    with open(os.path.join(path_test , "rawdata.pkl"), "rb") as file:
        totdata = pickle.load(file)
        # dims = totdata[0]
        # n = totdata[1]
        dataset = totdata[2]

    # size = n^dims 

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

    plt.show()