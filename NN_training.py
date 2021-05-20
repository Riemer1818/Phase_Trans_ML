# # -*- coding: utf-8 -*-
# """
# Created on Thu Apr 29 17:14:38 2021

# @author: jochem
# """


import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle
import os
import sys
from functions.functions import *


if __name__ == "__main__":

    dirname = sys.argv[1]

    dir = "./data"    
    path = os.path.join(dir, dirname)
    print("using: ", path, " as training inputfile")

    with open(os.path.join(path , "rawdata.pkl"), "rb") as file:
        totdata = pickle.load(file)
        dims = totdata[0]
        n = totdata[1]
        dataset = totdata[2]

    size = n^dims 

    dataset = np.concatenate(dataset) # nu hebben we een lijst van shape (5000,2) dus 5000 lijsten van de vorm [temp, grid]

    number_of_training_data = 30

    shape = [size,40,2]

    weights     = [np.random.uniform(-0.1,0.1,(shape[i],shape[i+1])) for i in range(len(shape)-1)]
    bias       = [np.random.uniform(-1,1,(shape[i+1])) for i in range(len(shape)-1)]

    nn = NeuralNetwork(shape, weights, bias, dataset, number_of_training_data) #eerste optie [200 ,50 , 30]
    
    nn.Desired_Out()
    foutmarge_training_data = []
    foutmarge_ongeziene_data = []
    weight_aanpas_groote = []
    nfactor_lijst = []

        
    nfactor = -3
    for k in range(3000):
        if k%10 == 0:
            foutmarge_ongezien = nn.test_ongeziene_data()
            foutmarge_training_data.append(nn.test_op_alle_traindata())
            foutmarge_ongeziene_data.append(foutmarge_ongezien)
            nfactor  = learning_rate_function(foutmarge_ongezien)
            nfactor_lijst.append(nfactor)

        if k %100 == 0:
            print(k)
            print(nfactor)
            print('training',nn.test_op_alle_traindata())
            print('ongezien',nn.test_ongeziene_data())
            print(nn.layer[-1])
            
        nn.feedforward(normaal = 0)
        nn.backprop(10 ** nfactor)
        
    # plt.figure('weight_groote & learning_rate')
    # plt.plot(weight_aanpas_groote, label ='weightgroote')  
    # plt.plot(nfactor_lijst, label = 'learning_rate')   
    # plt.legend()
    # plt.show()  
     
    # plt.figure('fout plot')
    # plt.plot(foutmarge_training_data, label = 'fout op trainingdata')  
    # plt.plot(foutmarge_ongeziene_data, label = 'fout op alledata inclusie ongeziene data')
    # plt.legend()
    # plt.plot(np.zeros(len(foutmarge_ongeziene_data))) # 0 lijn 
    # plt.xlabel('epochs')
    # plt.ylabel('genormaliserede fout')
    # plt.show() 
        
    np.save(os.path.join(path, 'weights.npy'), nn.weight)
    np.save(os.path.join(path, 'bias.npy'), nn.bias)
