# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:14:38 2021

@author: jochem
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
#import pandas_datareader as pdr
import timeit
np.random.seed(1)


"""--------------------------------------------------------------------------------------------------------""" 

def recover_graph(verhoudingen,beginwaarde):
    new_data = np.zeros(len(verhoudingen))
    new_data[0] = beginwaarde
    for i in range(len(verhoudingen)-1):
        new_data[i+1] = new_data[i]* verhoudingen[i+1]
        
    return new_data     
        

def sigmoid(X):
    """Sigmoid Function."""
    return  1/(1+np.exp(-X))

def diff_sigmoid(X):
    """Afgeleiden Sigmoid Function."""
    return (X)*(1.0-(X))


def ReLU(x, factor = 0.15):
    """Relu Function."""
    y1 = ((x > 0) * x)                                            
    return y1

def ReLU_leaky(x, factor = 0.15):
    """Relu Function."""
    y1 = ((x > 0) * x)                                            
    y2 = ((x <= 0) * x * factor)                                         
    leaky_way = y1 + y2  
    return leaky_way

def diff_ReLU(x):
    y1 = ((x > 0) * 1)                                                                                 
    return y1

def diff_ReLU_leaky(x,factor = 0.15):
    y1 = ((x > 0) * 1)                                            
    y2 = ((x <= 0) * factor)                                         
    leaky_way = y1 + y2
    return leaky_way


def cap(lijst, maxiwaarde):
    lijst= np.array(lijst)
    x = lijst > maxiwaarde
    y = lijst - lijst*x + maxiwaarde*x
    x = lijst < -maxiwaarde
    y = y - y*x - maxiwaarde*x
    return y

def softmax(output):
    totaal = sum(np.e ** output)
    return np.e ** output/totaal

class NeuralNetwork:
    """Een neuraal Netwerk."""

    def __init__(self, w,b, data, number_of_training_data,Tc): #shape [784,16,16,10] data is de volledige input, input_size
        
        # self.beginwaardes   = [i[1] for i in data]
        self.data_for_DO    = data
        self.full_data      = np.array([np.reshape(i[1],-1) for i in data])
        self.traing_data    = self.full_data 
        
        
        self.weight     = w
        self.bais       = b
        self.Tc         = Tc
        
        self.shape      = [np.shape(i)[0] for i in self.weight]
        self.shape.append(np.shape(self.bais[-1])[0])
        
        self.input_size  = int(self.shape[0])
        self.output_size = int(self.shape[-1])

        self.weight_aanpas_groote = 0
    
    def Desired_Out(self):
        """Deze function defineerd wat de desired output word."""
        self.DO_all = []   #aantal output neuronen
        kritische_temp = self.Tc
        for i in self.data_for_DO:
            if i[0]< kritische_temp:
                self.DO_all.append([1,0])
            elif i[0] >= kritische_temp :
                self.DO_all.append([0,1])
        # print(self.DO_all)
                
            

    def feedforward(self,aantal = 30 ,normaal= 0): 
        """Feedforward van data."""
        "normaal 0 is gwn feedforward"
        "normaal 1 test op alle trainings data"
        "normaal 2 test op alle data ook nog nooit geziene data"
        
        """instellen random gekozen data."""
        """Er zijn twee opties of random een bathc of alles dat alles is om te checken
        let hier bij op dat er twee opties zijn voor random of alles van je ranodm training set of van je daadwerkelijk data"""
        if normaal == 0:
            aantal = aantal
        elif normaal == 1:
            aantal = len(self.traing_data)
        elif normaal == 2:
            aantal = len(self.full_data)
        else:
            print('something went wrong with normaal parameter')
            
        self.index = np.zeros(aantal,dtype = int) #dit is voor de desired output
        shape = self.shape[0] #shape input data
        self.input = np.zeros([aantal,shape])
            

        if normaal == 0:
            'random choice out of trainingdata'
            for i in enumerate(np.random.choice(len(self.traing_data),size = aantal)):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.full_data[i[1]][0:self.input_size] 
        elif normaal == 1:
            'all trainingdata'
            for i in enumerate(range(len(self.traing_data))):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.full_data[i[1]][0:self.input_size] 
        elif normaal == 2:
            for i in enumerate(range(len(self.full_data))):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.full_data[i[1]][0:self.input_size] 
            
                
        self.DO = np.zeros([aantal,self.shape[-1]])  #creeer lijst met aantal batch met lijsten van lengte output
        for i in enumerate(self.index):
            self.DO[i[0]] = self.DO_all[i[1]]
        
            
        """Daadwerkelijke feedforword."""
        combi = zip(self.weight, self.bais) 
        self.layer = [self.input]                     #we willen elke layer opslaan
        invoer = self.input
        for i in enumerate(combi):
            if i[0] == len(self.weight)-1:
                layer = ReLU_leaky(np.add(np.dot(invoer, i[1][0]),i[1][1]))
                layer = np.array([softmax(i) for i in layer])
                # print(layer)
                # layer = sigmoid(np.add(np.dot(invoer, i[1][0]),i[1][1]))
            else:
                
                layer = ReLU_leaky(np.add(np.dot(invoer, i[1][0]),i[1][1]))
                # layer = sigmoid(np.add(np.dot(invoer, i[1][0]),i[1][1]))
            self.layer.append(layer)
            invoer = layer
   

    """raar dat ik ouput gebruik ipv z """


    def backprop(self,learning_rate):
        """Backprop."""
        learning_rate = learning_rate
        pre_error = 2*(self.DO - self.layer[-1]) #[ [] , [] , [] ]
        """gebruik andere costfucntie"""
        for i in range(len(self.layer)-1):        #layer is van de feedforward
            output = self.layer[-i-1] # pak eerst output
            


            # error = pre_error * diff_sigmoid(output)  #[[cost voor 1 inputs], [ cost voor 2de inputs]]
            error = pre_error * diff_ReLU_leaky(output)

            samen_W = zip(error,self.layer[-i-2])
            
            #weights
            d_weight = [[j*k[0] for j in k[1]] for k in samen_W]

            d_weight = (np.sum(d_weight,0)/(len(error))) * learning_rate #gemiddelde
           

            #baises 
            d_bais = (np.sum(error,0)/(len(error))) * learning_rate  #gemiddelde
            
            #update zodat ze niet te groot worden
            d_weight    = cap(d_weight,0.01)
            d_bais      = cap(d_bais,0.2)
            # print(d_weight)
            self.weight[-i-1]  += d_weight
            self.bais[-i-1]     += d_bais
            
            self.weight_aanpas_groote = np.sum(d_weight)/len(d_weight) * 1000


            if i != len(self.layer)-2: #laatste hoeft niet
                pre_error = [np.dot(laag_error,self.weight[-1-i].T) for laag_error in error]
                
   
    def random_weight_bais(self,highed):
        for i in enumerate(self.weight):
            self.weight[i[0]] += np.random.uniform(-highed,highed,np.shape(i[1]))
        for i in enumerate(self.bais):
                self.bais[i[0]] += np.random.uniform(-highed,highed,np.shape(i[1]))
    
    def test_op_alle_traindata(self):
        self.feedforward(normaal = 1)
        output = self.layer[-1]
        combi = zip(output,self.DO)
        som = 0
        for i in combi:
            som += np.sum(np.abs(np.abs(i[0])-np.abs(i[1])))
        return -som/len(output) #genormaliseerd
    
    def test_ongeziene_data(self):
        self.feedforward(normaal = 2)
        output = self.layer[-1]
        combi = zip(output,self.DO)
        som = 0
        for i in combi:
            som += np.sum(np.abs(np.abs(i[0])-np.abs(i[1])))
        return -som/len(output) #genormaliseerd
    
    def conclusieT(self):
         self.feedforward(normaal = 2)
         output = self.layer[-1]
         # print(self.data_for_DO[0:50])
         T  = []
         y1 = []
         y1_std = []
         y2 = []
         y2_std = []
         lengte = int(len(output)/50)
         "50 want iedere T heeft 50 grids"
         for i in range(0,lengte):
               sub_output = output[i*50:(i+1)*50]
               lijsty1 = [i[0] for i in sub_output]
               y1.append(np.mean(lijsty1))
               y1_std.append(np.std(lijsty1))
               lijsty2 = [i[1] for i in sub_output]
               y2.append(np.mean(lijsty2))
               y2_std.append(np.std(lijsty2))
               T.append(self.data_for_DO[i*50][0])
         return T,y1,y1_std,y2,y2_std
    
         
def learning_rate_function(X,grens = 0.1): #  kies je factor je eerste begin foutmarge, factor/2 ^ 2 is je sig
    """wil proberen de eerste fout op 4sigma te zetten dat lijkt logisch omdat dat nog redelijk fout is dan dan zit
    bij de minimale fout .
    kies je sigma dus eerstfout /4"""
    sigma= 3
    learning_rate = -1* np.e ** -((X/sigma) ** 2) - 2  #chech in geogabra mooie functie die -4 is laagste learning rate
    if -grens < X < grens:
        return -5
    else:
        return learning_rate

def potentiale(x, grens = 1.5, minimum = -10, verhouding = 50):
    if x < grens and x > -grens:
        return -10
    else:
        return -1/(np.abs(x)**2/verhouding) - 5

def history_learningrate(lijst_foutmarge, new_fout,stimulans = 0.3, lengte_geschiedenis= 30):
    y = 0
    if len(lijst_foutmarge)< lengte_geschiedenis+1:
        return 0
    else:
        for i in range(1,lengte_geschiedenis+1):
            y += lijst_foutmarge[-i]

        y = y/lengte_geschiedenis #gemiddelde fout

        # print('fout', np.abs(new_fout-y),  0.20 * y)
        if np.abs(new_fout-y) < 0.04 * np.abs(y): #tienprocent foutmarge
            # print('history')
            return stimulans
        else:
            return 0
        
L = 20
dims = 2

size = L**dims

shape = [size,40,2]
#%%
Tk = 2.27
Tcs = list(np.linspace(0.001,2*Tk, 25))
#Tcs.append(2.27)
print(Tcs)
trained_accuracies = []
trainedcorrect_accuracies = []
test_accuracies = []
testcorrect_accuracies = []
#%%
for i in range(len(Tcs)):

    w     = [np.random.uniform(-0.1,0.1,(shape[i],shape[i+1])) for i in range(len(shape)-1)]
    b     = [np.random.uniform(-1,1,(shape[i+1])) for i in range(len(shape)-1)]
    
    #w = ([np.load('f_beste_versie2_w.npy', allow_pickle=True)])[0]
    #b = ([np.load('f_beste_versie2_b.npy', allow_pickle=True)])[0]
    dataset = np.load('f_train_data.npy', allow_pickle=True)
    dataset = np.concatenate(dataset) # nu hebben we een lijst van shape (5000,2) dus 5000 lijsten van de vorm [temp, grid]
    
    number_of_training_data = 30
    
    nn = NeuralNetwork(w,b,dataset, number_of_training_data,Tcs[i]) #eerste optie [200 ,50 , 30]
    nn.Desired_Out()
    foutmarge_ongeziene_data = []
    weight_aanpas_groote = []
    nfactor_lijst = []
    
    nfactor = -3
    for k in range(10000):
        if k%10 == 0:
            foutmarge_ongezien = nn.test_ongeziene_data()
            foutmarge_ongeziene_data.append(foutmarge_ongezien)
            nfactor  = learning_rate_function(foutmarge_ongezien)
            nfactor_lijst.append(nfactor)
        if k %100 == 0:
            print("k = " + str(k))
            print(nfactor)
            print('error',nn.test_ongeziene_data())
            print("Tc = " + str(Tcs[i]) + " i = " + str(i))
            
        nn.feedforward(normaal = 0)
        nn.backprop(10 ** nfactor)        
    
    trained_accuracies.append(nn.test_ongeziene_data())
    
    trained_w = nn.weight
    trained_b = nn.bais
    
    nn = NeuralNetwork(trained_w,trained_b,dataset, number_of_training_data,2.27)
    nn.Desired_Out()
    trainedcorrect_accuracies.append(nn.test_ongeziene_data())
    
    dataset = np.load('f_test_data.npy', allow_pickle=True)
    dataset = np.concatenate(dataset)
    
    nn = NeuralNetwork(trained_w,trained_b,dataset, number_of_training_data,Tcs[i]) #eerste optie [200 ,50 , 30]
    nn.Desired_Out()
    test_accuracies.append(nn.test_ongeziene_data())
    
    nn = NeuralNetwork(trained_w,trained_b,dataset, number_of_training_data,2.27) #eerste optie [200 ,50 , 30]
    nn.Desired_Out()
    testcorrect_accuracies.append(nn.test_ongeziene_data())
    

#%%

plt.scatter(Tcs,trained_accuracies)
plt.show()

plt.scatter(Tcs,trainedcorrect_accuracies)
plt.show()

plt.scatter(Tcs,test_accuracies)
plt.show()

plt.scatter(Tcs,testcorrect_accuracies)
plt.show()

#%%

np.save('Tcs,n=25,e=10000',Tcs)
np.save('trained_accuracies,n=25,e=10000',trained_accuracies)
np.save('trainedcorrect_accuracies,n=25,e=10000',trainedcorrect_accuracies)
np.save('test_accuracies,n=25,e=10000',test_accuracies)
np.save('testcorrect_accuracies,n=25,e=10000',testcorrect_accuracies)

















