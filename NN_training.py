# # -*- coding: utf-8 -*-
# """
# Created on Thu Apr 29 17:14:38 2021

# @author: jochem
# """


import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle
import os, sys
from functions.functions import *


if __name__ == "__main__":

    # dirname = sys.argv[1]
    dirname = '../data/anti_2D_20grid_10000itir_100step'
    path = ''     #opslaan data
    print("using: ", dirname, " as training input directory")
    print(os.path.dirname(os.path.realpath(__file__)))

    dims, n, number_of_training_data, totdata = unpickle_dir(dirname)

    dataset = np.concatenate(totdata)

    # dataset = np.concatenate(totdata) # nu hebben we een lijst van shape (5000,2) dus 5000 lijsten van de vorm [temp, grid]

    size = n^dims 
    shape = [size,40,2]
    batch_size = 40
    weights     = [np.random.uniform(-0.1,0.1,(shape[i],shape[i+1])) for i in range(len(shape)-1)]
    bias       = [np.random.uniform(-1,1,(shape[i+1])) for i in range(len(shape)-1)]

    nn = NeuralNetwork(shape, weights, bias, dataset, number_of_training_data) #eerste optie [200 ,50 , 30]
    
    nn.Desired_Out()
    foutmarge_training_data = []
    foutmarge_ongeziene_data = []
    weight_aanpas_groote = []
    nfactor_lijst = []

        
    nfactor = -1
    for k in range(3000):
        if k%10 == 0:
            foutmarge_ongezien = nn.test_ongeziene_data()
            foutmarge_training_data.append(nn.test_op_alle_traindata())
            foutmarge_ongeziene_data.append(foutmarge_ongezien)
            # nfactor  = learning_rate_function(foutmarge_ongezien)
            # nfactor_lijst.append(nfactor)

        if k %100 == 0:
            print(k)
            print(nfactor)
            print('training',nn.test_op_alle_traindata())
            print('ongezien',nn.test_ongeziene_data())
            print(nn.layer[-1])
            
        nn.feedforward(aantal = batch_size, normaal = 0)
        nn.backprop(10 ** nfactor)
        
    np.save(os.path.join(path, 'weights.npy'), nn.weight)
    np.save(os.path.join(path, 'bias.npy'), nn.bias)