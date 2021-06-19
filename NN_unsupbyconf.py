# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:14:38 2021

@author: jochem
"""

# karel

import datetime
import matplotlib.pyplot as plt
import numpy as np
#import pandas_datareader as pdr
#from functions.functions import *
import timeit
import sys, os
import pickle 

np.random.seed(1)

def ReLU_leaky(x, factor = 0.15):
    """Relu Function."""
    y1 = ((x > 0) * x)                                            
    y2 = ((x <= 0) * x * factor)                                         
    leaky_way = y1 + y2  
    return leaky_way

def softmax(output):

    output = output - max(output)[0]
    # if (max(np.abs(output))[0])/2 > 10:
    totaal = sum(np.e ** output)
    return np.e ** output/totaal

def Jacobian_softmax(output):
    N = len(output)
    s1 = output[0][0]
    s2 = output[1][0]
    jacobian = np.array([[-s2,s2],[s1,-s1]])

    return jacobian
                                                            
def Jacobian_ReLU_leaky(x,factor = 0.15):
    y1 = ((x > 0) * 1)                                            
    y2 = ((x <= 0) * factor)                                         
    leaky = y1 + y2
    N = len(leaky)
    matrix = np.zeros((N,N))
    for i in range(N):
        matrix[i][i] = leaky[i]
    return matrix                                                           
            
def makematrix(inp,lengte):
    matrix = []
    for i in range(lengte):
        matrix.append(inp)
        
    return matrix
    
class NeuralNetwork:
    """Een neuraal Netwerk."""

    def __init__(self, shape, weights, bias, train_data, test_data, Tk = None): #shape [784,16,16,10] data is de volledige input, input_size
        
        self.shape          = shape
        
        # self.beginwaardes   = [i[1] for i in data]
        self.input_size     = int(shape[0])
        self.output_size    = int(shape[-1])
        #100 data per temp
        self.train_data = train_data
        self.test_data = test_data
        
        'wat in de volgende 9 regels gebeurt is vrij gevoelig qua data input.'
        'dat kan dus beteken dat hier wat aangepast moet worden als we het op de surver draaien'
        'heb het nu zo gemaakt dat het werkt voor een georderde lijst van [[temp,grid],etc]'
        self.DO_train    = self.Desired_Out(self.train_data)
        self.DO_test     = self.Desired_Out(self.test_data)
        
        self.temp_train  = [i[0] for i in self.train_data]
        self.temp_test   = [i[0] for i in self.test_data]

        self.weight     = weights
        self.bias       = bias
        
        self.Tk = Tk

    def Desired_Out(self,data):
        """Deze function defineerd wat de desired output word."""
        DO = []   #aantal output neuronen
        kritische_temp = Tk
        for i in data:
            if i[0]< kritische_temp:
                DO.append([[1],[0]])
            elif i[0] >= kritische_temp :
                DO.append([[0],[1]])
                
        return DO
        # print(self.DO_all)
                
    def feedforward(self, aantal = 100 ,normaal= 0): 
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
            aantal = len(self.train_data)
        elif normaal == 2:
            aantal = len(self.test_data)
        else:
            print('something went wrong with normaal parameter')
            
        self.index = np.zeros(aantal,dtype = int) #dit is voor de desired output
        shape = self.shape[0] #shape input data
        self.input = np.zeros([aantal,shape,1])


        if normaal == 0:
            'random choice out of trainingdata'
            for i in enumerate(np.random.choice(len(self.train_data),size = aantal)):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.train_data[i[1]][0:self.input_size] 
        elif normaal == 1:
            'all trainingdata'
            for i in enumerate(range(len(self.train_data))):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.train_data[i[1]][0:self.input_size] 
        elif normaal == 2:
            'alle test data'
            for i in enumerate(range(len(self.test_data))):
                self.index[i[0]]= int(i[1])                                    #sla index op zodat de DO matched
                self.input[i[0]] = self.test_data[i[1]][0:self.input_size] 
            
        
        shapeDO = np.shape(self.DO_train[0]) #gwn shape van de DO output, dit is zo nodig
        self.DO = np.zeros([aantal,shapeDO[0],shapeDO[1]])  
        if normaal != 2:
            #creeer lijst met aantal batch met lijsten van lengte output
            for i in enumerate(self.index):
                self.DO[i[0]] = self.DO_train[i[1]]
        else:
            for i in enumerate(self.index):
                self.DO[i[0]] = self.DO_test[i[1]]
        
            
        """Daadwerkelijke feedforword."""
        combi = zip(self.weight, self.bias) 
        self.layer = [[self.input,self.input]]                  #we willen elke layer opslaan
        invoer = self.input
        for i in enumerate(combi):
            if i[0] == len(self.weight)-1:          #laatste layer
                z = np.add((i[1][0] @ invoer),i[1][1])
                layer = z
                'doe geen Relu meer voor mijn softmax'
                # layer = ReLU_leaky(z)
                layer = np.array([softmax(i) for i in layer])

                # layer = sigmoid(np.add(np.dot(invoer, i[1][0]),i[1][1]))
            else:

                z = np.add((i[1][0]@invoer),i[1][1])
                layer = ReLU_leaky(z)
                
                # layer = sigmoid(np.add(np.dot(invoer, i[1][0]),i[1][1]))

            self.layer.append([z,layer]) #z is raw output, layer is met Activtion er over heen
            invoer = layer
   

    """raar dat ik ouput gebruik ipv z """


    def backprop(self,learning_rate):
        """Backprop."""

        learning_rate = learning_rate

        pre_softmax = [Jacobian_softmax(i) for i in self.layer[-1][1]] #2,2 shape

        DO = [-i.T[0] for i in self.DO] #deze min deed alles fantastisch omdat het logg

        combi = zip(DO ,pre_softmax)
        
        pre_error = [i[0]@i[1] for i in combi] 
        self.preloss = pre_error
        self.update = []
        """gebruik andere costfucntie"""
        for i in range(len(self.layer)-1):        #layer is van de feedforward
            inp = self.layer[-i-2][1] #dit is zonder relu met relu, pak eerst input van uitachter gerekend
            lengte = len(self.preloss[0])  #gewoon 1 pakken voor de dimensie
            inp = np.array([makematrix(i.T[0], lengte) for i in inp])
            combi = zip(self.preloss, inp)
            Update_W = [np.multiply(np.reshape(j[0],(-1,1)),j[1])for j in combi]


    
            Update_W = (sum(Update_W,0)/len(Update_W)) * learning_rate #normaliseer over batch size


            self.update.append(Update_W)
            self.weight[-i-1]  += Update_W
            # print(Update_W)

            """for bais we have DL(b) = I identiteit matrix"""

            Update_B = (np.sum(self.preloss,0)/len(pre_error)) * learning_rate
            Update_B = np.reshape(Update_B,(-1,1))
            
            'omdat pre_erro shape (1,2) heeft maar moet (2,1) worden moeten we transpose in bais.'
            
            self.bias[-i-1]     += Update_B
           
            "DL(I) berekenen zodat we de volgende update voor W en B kunnen berekenen"
            "voor de laatste laag hoeft dit niet meer gedaan te worden want daar doen we niks meer mee."
            if i != len(self.layer)-2:
                

                DL = [self.weight[-i-1] for j in pre_error]
                pre_ReLU = [Jacobian_ReLU_leaky(i) for i in self.layer[-i-2][0]]
                combi = zip(self.preloss,DL,pre_ReLU)
                self.preloss = [j[0]@j[1]@j[2] for j in combi]
    
    def testen_nn(self, normaal): #normaal = 1 of 2
        self.feedforward(normaal = normaal)
        output = self.layer[-1][1]
        combi = zip(output,self.DO)
        som = 0
        for i in combi:
            som += np.sum(np.abs(np.abs(i[0])-np.abs(i[1])))
        return -som/len(output) #genormaliseerd

#%%
   
#train_dirname = sys.argv[1]
train_dirname = 'C:/Users/karel/Documents/UCU/SEM8/Complex_Systems_Project/Data/train_data_ML/normal_2D_20grid_30itir_100step'
print("using: ", train_dirname, " as training input directory")

#test_dirname = sys.argv[2]
test_dirname = 'C:/Users/karel/Documents/UCU/SEM8/Complex_Systems_Project/Data/test_data_ML/normal_2D_20grid_30itir_100step'
print("using: ", test_dirname, " as test input directory")
    
traindata = np.load(os.path.join(train_dirname, 'data_normal_2D_20grid_30itir_100step.npy'), allow_pickle=True)
traindata = traindata[traindata[:,0].argsort()]
#train_totdata = np.concatenate(traindata)

testdata = np.load(os.path.join(test_dirname, 'data_normal_2D_20grid_30itir_100step.npy'), allow_pickle=True)
testdata = testdata[testdata[:,0].argsort()]
#test_totdata = np.concatenate(testdata)

#%%

print(traindata)

#%%

epochs = 100
steps = 3	
dims = 2
n = 20
size = n**dims 

traindata    = np.array([np.reshape(i[1],(size,1)) for i in traindata]) 
testdata     = np.array([np.reshape(i[1],(size,1)) for i in testdata])

#%%

#%%
def out_dirnamer(Tk, epochs, steps, train_dirname):
    output_dirname = 'grid'+str(n) + '_' + 'Tk' + str(Tk)+'_'+ 'epochs'+str(epochs)+'_'+'steps'+str(steps)
    os.mkdir(output_dirname)
    return output_dirname

Tk = 2.27
#Tk = 4.5
#Tk = 6.86

out_dirname = out_dirnamer(Tk, epochs, steps, train_dirname)

shape = [size,4,2]

Tks = list(np.linspace(0.001,2*Tk, steps))

trained_accuracies = []
test_accuracies = []

for i in range(len(Tks)):
    weights     = [np.random.uniform(-0.1,0.1,(shape[i],shape[i+1])) for i in range(len(shape)-1)]
    bias        = [np.random.uniform(-1,1,(shape[i+1])) for i in range(len(shape)-1)]
    
    nn = NeuralNetwork(shape, weights, bias, traindata, testdata, Tks[i]) 
    #nn.Desired_Out()
    
    foutmarge_traindata = []
    weight_aanpas_groote = []
    nfactor_lijst = []
    
    nfactor = -1
    
    for k in range(epochs):
        if k %10 == 0:
            print("k = " + str(k))
            print("Tk = " + str(Tks[i]) + " i = " + str(i))
            
        nn.feedforward(normaal = 0)
        nn.backprop(10 ** nfactor)        
    
    trained_accuracies.append(nn.testen_nn(normaal=1))
    test_accuracies.append(nn.testen_nn(normaal=2))

plt.scatter(Tks,trained_accuracies)
plt.savefig(os.path.join(out_dirname, 'trained_accuracies.png'))

plt.scatter(Tks,test_accuracies)
plt.savefig(os.path.join(out_dirname, 'test_accuracies.png'))

np.save(os.path.join(out_dirname, 'Tks'), Tks)
np.save(os.path.join(out_dirname, 'trained_accuracies'), trained_accuracies)
np.save(os.path.join(out_dirname, 'test_accuracies'), test_accuracies)

















