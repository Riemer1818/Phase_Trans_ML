# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:14:38 2021

@author: jochem
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
#import pandas_datareader as pdr
from functions.functions import *
import timeit
import sys, os
import pickle 

np.random.seed(1)


if __name__ == "__main__":
    
    Tk = 2.27
    epochs = 10000
    steps = 25

    train_dirname = sys.argv[1]
    print("using: ", train_dirname, " as training input directory")
    test_dirname = sys.argv[2]
    print("using: ", test_dirname, " as test input directory")

    train_totdata = []
    number_of_training_data = 0 
    test_totdata = []
    number_of_testing_data = 0

    for filename in os.listdir(train_dirname):
        if filename.endswith('.pkl'):
            f = os.path.join(train_dirname, filename)
            with open(f, "rb") as file:
                totdata = pickle.load(file)
                dims = totdata[0]
                n = totdata[1]
                dataset = totdata[2]
                number_of_training_data += 1

        else:
            pass

        totdata.append(dataset)

    train_totdata = np.concatenate(totdata) # nu hebben we een lijst van shape (5000,2) dus 5000 lijsten van de vorm [temp, grid]

    for filename in os.listdir(test_dirname):
        if filename.endswith('.pkl'):
            f = os.path.join(test_dirname, filename)
            with open(f, "rb") as file:
                totdata = pickle.load(file)
                dataset = totdata[2]
                number_of_testing_data += 1

        else:
            pass

        totdata.append(dataset)

    test_totdata = np.concatenate(totdata)

    size = n^dims 
    shape = [size,40,2]

    Tks = list(np.linspace(0.001,2*Tk, steps))

    trained_accuracies = []
    test_accuracies = []

    for i in range(len(Tks)):
        weights     = [np.random.uniform(-0.1,0.1,(shape[i],shape[i+1])) for i in range(len(shape)-1)]
        bias        = [np.random.uniform(-1,1,(shape[i+1])) for i in range(len(shape)-1)]
        
        nn = NeuralNetwork(shape, weights, bias, train_totdata, number_of_training_data, Tks[i]) 
        nn.Desired_Out()
        
        foutmarge_ongeziene_data = []
        weight_aanpas_groote = []
        nfactor_lijst = []
        
        nfactor = -3
        
        for k in range(epochs):
            if k%10 == 0:
                foutmarge_ongezien = nn.test_ongeziene_data()
                foutmarge_ongeziene_data.append(foutmarge_ongezien)
                nfactor  = learning_rate_function(foutmarge_ongezien)
                nfactor_lijst.append(nfactor)

            if k %100 == 0:
                print("k = " + str(k))
                print(nfactor)
                print('error',nn.test_ongeziene_data())
                print("Tk = " + str(Tks[i]) + " i = " + str(i))
                
            nn.feedforward(normaal = 0)
            nn.backprop(10 ** nfactor)        
        
        trained_accuracies.append(nn.test_ongeziene_data())
        
        trained_w = nn.weight
        trained_b = nn.bais
        
        nn = NeuralNetwork(trained_w, trained_b, test_totdata, number_of_training_data, Tks[i]) #eerste optie [200 ,50 , 30]
        nn.Desired_Out()
        test_accuracies.append(nn.test_ongeziene_data())

    plt.scatter(Tks,trained_accuracies)
    plt.show()

    plt.scatter(Tks,test_accuracies)
    plt.show()

    np.save(os.path.join(test_dirname, 'Tks'), Tks)
    np.save(os.path.join(test_dirname, 'trained_accuracies'), trained_accuracies)
    np.save(os.path.join(test_dirname, 'test_accuracies'), test_accuracies)

















